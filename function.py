# library functions ----------------------------------------------------------------------------------------
from pandas import DataFrame, concat

from numpy import isnan, mean

from plotnine import ggplot, aes, labs, position_dodge, after_stat
from plotnine import geom_bar, geom_boxplot, geom_histogram, geom_col, geom_text, facet_wrap
from plotnine import scale_x_continuous, scale_y_continuous, scale_x_discrete, scale_y_discrete, scale_fill_manual, coord_flip, coord_cartesian
from plotnine import theme_light, theme, element_blank, element_text, element_rect

from sklearn.impute import KNNImputer

# layout-ncol: 2

# Outliers analysis functions ------------------------------------------------------------------------------
def get_outlier (df, var, limit, typ = "both"):
    """
    parameter
    ---------
    df    [DataFrame]
    var   [int, float64] A numerical variable from the data `df`.
    limit [1.5, 3] The type of limit to use when calculating the outlier. (3) to get extreme outliers.
    typ   [string] The type of output to return.
    
    return
    ------
    when typ = 'lower' retur
    """
    if limit not in [1.5, 3]:
        raise ValueError(f"argument `limit` must be 1.5 or 3 and not {limit}")
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_outlier = Q1 - limit * IQR
    upper_outlier = Q3 + limit * IQR
    
    if typ == "lower":
        return lower_outlier
    elif typ == "upper":
        return upper_outlier
    elif typ == "both":
        return [lower_outlier , upper_outlier]
    else:
        raise ValueError(f"argument `typ` must be any of 'lower', 'upper' or 'both' and not {typ}")
        
        
        
def tukeys (df, var, typ = "insight"):
    """
    parameter
    ---------
    df    [DataFrame]
    var   [int, float64] A numerical variable from the data `df`.
    typ   [string] The type of output to return.
    
    return
    ------
    """
    lower_inner_fance  = get_outlier(df, var, 1.5, "lower")
    upper_inner_fance  = get_outlier(df, var, 1.5, "upper")
    
    lower_outter_fence = get_outlier(df, var, 3, "lower")
    upper_outter_fence = get_outlier(df, var, 3, "upper")
    
    outlier_prob = []
    outlier_poss = []
    
    for index, i in enumerate(df[var]):

        if i <= lower_outter_fence or i >= upper_outter_fence:
            outlier_prob.append(index)
        if i <= lower_inner_fance or i >= upper_inner_fance:
            outlier_poss.append(index)
    
    if typ == "inner":
        return outlier_poss
    elif typ == "outer":
        return outlier_prob
    elif typ == "insight":
        print(f'Inner Fence Index : {outlier_poss}\n\nOuter Fence Index : {outlier_prob}')
    else:
        raise ValueError(f"argument `typ` can ba any of 'inner', 'outer' or 'insight' and not {typ}")
      
    
    
        
# filter rows with outliers --------------------------------------------------------------------------------
def filter_outlier(df, var, typ):
    """
    parameter
    ---------
    df  [DataFrame]
    var [int, float64] A numerical variable from the data `df`.
    typ [string] The type of output to return (passed to the tukeys function).
                 any of 'inner' or 'outer'
    
    return
    ------
    """
    from pandas import DataFrame
    
    return df.iloc[tukeys(df, var, typ), :]




# Outlier Imputation ----------------------------------------------------------------------------------------
def imputation (df, var, new_col_name = None, lower_metric = None, upper_metric = None, lim = 3):
    """
    df            [DataFrame]
    var           [int, float64] A numerical variable from the data `df` with outliers.
    new_col_name  [string] The name of the column to call the imputed variabel `var`.
    lower_metric, [int, float64] A value to impute inplace of outlier values.
    upper_metric
    lim           [1.5, 3] The 
    """
    from numpy import where
    from pandas import DataFrame
    
    var_outlier = get_outlier(df, var, lim)
            
    if upper_metric is not None and lower_metric is not None:
        imp = df[var]
        imp = where(imp < var_outlier[0], lower_metric, imp)
        imp = where(imp > var_outlier[1], upper_metric, imp)
        if new_col_name is None:
            df[var] = imp
        else:
            df[new_col_name] = imp
            
    elif lower_metric is not None:
        imp = where(df[var] < var_outlier[0], lower_metric, df[var])
        if new_col_name is None:
            df[var] = imp
        else:
            df[new_col_name] = imp
            
    elif upper_metric is not None:
        imp = where(df[var] > var_outlier[1], upper_metric, df[var])
        if new_col_name is None:
            df[var] = imp
        else:
            df[new_col_name] = imp
    
    return df[var] if new_col_name is None else df[new_col_name] 




# Missing value Imputation ---------------------------------------------------------------------------------------
def cat_imputation(df, cat_var, imp_var, imp_fun, impute_df = None, return_typ = "float"):
    f_tbl = df.copy()
    
    imp_df = f_tbl.groupby(cat_var)[imp_var].agg(imp_fun).reset_index()
    
    for obs in imp_df[cat_var].tolist():
        imp_value = imp_df.query(f"{cat_var} == '{obs}'")[imp_var].values
        if isnan(imp_value):
            imp_value = mean(f_tbl[imp_var])
            
        imp_value = float(imp_value) if return_typ == "float" else int(imp_value)

        if impute_df is None:
            f_tbl.loc[(f_tbl[imp_var].isna()) & (f_tbl[cat_var] == obs), imp_var] = imp_value
            tbl = f_tbl
        else:
            impute_df.loc[(impute_df[imp_var].isna()) & (impute_df[cat_var] == obs), imp_var] = imp_value
            tbl = impute_df
    return tbl




# Data cleaning and Feature Engineering Functions ----------------------------------------------------------------

# creating particulate matter (PM2.5) category

def add_pmCategory(df, ordinal=False):
    """
    df [DataFrame] 
    ordinal [bool] create an ordinal encoding.
    """
    def PM_category(df, pm_col, aqi_col, encode_ordinal = ordinal):
        if encode_ordinal:
            df.loc[df[pm_col] <= 1000,  aqi_col] = 0
            df.loc[df[pm_col] <= 250.4, aqi_col] = 1
            df.loc[df[pm_col] <= 150.4, aqi_col] = 2
            df.loc[df[pm_col] <= 55.4,  aqi_col] = 3
            df.loc[df[pm_col] <= 35.4,  aqi_col] = 4
            df.loc[df[pm_col] <= 12,    aqi_col] = 5
            
            df[aqi_col] = df[aqi_col].astype("int64")
        else:
            df.loc[df[pm_col] <= 1000, aqi_col]  = "Hazardous"
            df.loc[df[pm_col] <= 250.4, aqi_col] = "Very Unhealthy"
            df.loc[df[pm_col] <= 150.4, aqi_col] = "Unhealthy"
            df.loc[df[pm_col] <= 55.4,  aqi_col] = "Unhealthy (SG)"
            df.loc[df[pm_col] <= 35.4,  aqi_col] = "Moderate"
            df.loc[df[pm_col] <= 12,    aqi_col] = "Good"
            
        return df
    
    df = PM_category(df, "Sensor1_PM2.5", "S1_AQI")
    df = PM_category(df, "Sensor2_PM2.5", "S2_AQI")
    
    return df



# Extracting Dates
def get_standAlone_dates(df, date_col, which = None):
    """
    parameter
    ---------
    df       [DataFrame]
    date_col [datetime64[ns]] variable to extract the values from.
    
    return
    ------
    pd.DataFrame containing all included variables
    """
    if which is None:
        df['hour']  = df[date_col].dt.hour
        df['day']   = df[date_col].dt.day
        df['month'] = df[date_col].dt.month
        df['year']  = df[date_col].dt.year
        df["day_of_year"]  = df[date_col].dt.dayofyear 
        df["week_of_year"] = df[date_col].dt.weekofyear
    else:
        dates_dict = {"hour" : df[date_col].dt.hour,
                      "day"  : df[date_col].dt.day,
                      "month": df[date_col].dt.month,
                      "year" : df[date_col].dt.year,
                      "day_of_year": df[date_col].dt.dayofyear,
                      "week_of_year": df[date_col].dt.weekofyear}
        for dates in which:
              df[str.title(dates)] = dates_dict[dates]
            
    return df


class add_attributes:
    def __init__(self, df, drop_nan_value=False, impute=False, fill_nan_value=True, add_aqi=True, add_date=True):
        self.df = df.copy()
        
        nan_var = df.isnull().sum() 
        self.nan_var = list(nan_var[nan_var > 0].index)
        self.drop_nan_value = drop_nan_value
        self.impute = impute
        self.fill_nan_value = fill_nan_value
        self.add_aqi = add_aqi
        self.add_date = add_date
        
    def drop_missing_value(self):
        f_df = self.df
        
        if self.drop_nan_value:
            self.df = f_df.dropna()
            return self.df
        else:
            return f_df
            
            
    def fill_missing_value(self, fill_fun="median", group_var=None):
        f_df = self.df
        
        if self.fill_nan_value:
            
            if self.nan_var != []:
                
                if group_var is None:
                    for var in self.nan_var:
                        if fill_fun == "median":
                            f_df[var] = f_df[var].fillna(f_df[var].median())
                        elif fill_fun == "mean":
                            f_df[var] = f_df[var].fillna(f_df[var].mean())
                            
                    return f_df
                
                else:
                    
                    for var in self.nan_var:
                        if fill_fun == "median":
                            f_df[var] = f_df[var].fillna(f_df.groupby(group_var)[var].transform('median'))
                        elif fill_fun == "mean":
                            f_df[var] = f_df[var].fillna(f_df.groupby(group_var)[var].transform('mean'))
                            
                    return f_df
                
            else:
                return f_df
            
        else:
            return f_df
                    
            
    def impute_knn(self, imp_weights="uniform", imp_metric="nan_euclidean", fill_fun = "median", group_var=None):
        f_df = self.df
        
        if self.impute:
            
            knn_imputer = KNNImputer(n_neighbors=7, weights=imp_weights, metric=imp_metric)
            
            if self.nan_var != []:
                
                for var in self.nan_var:
                    
                    f_df[[var]] = knn_imputer.fit_transform(f_df[[var]])
                    
                return f_df
            
            else:
                return f_df
            
        else:
            return f_df
     
    
    def add_air_quality_index(self, encode_ordinal = True):
        f_df = self.df
        
        if self.add_aqi:
            return add_pmCategory(f_df, ordinal = encode_ordinal)
        
        else:
            return f_df
        
        
    def add_period_variables(self, hour=False, day=False, month=False, year=False, dayofyear=False, weekofyear=False):
        f_df = self.df
        
        if self.add_date:
            
            period = dict(hour=hour, day=day, month=month, year=year, day_of_year=dayofyear, week_of_year=weekofyear)
            extract_period = [key for key, value in period.items() if value == True]
            
            if extract_period == []:
                f_df = get_standAlone_dates(df=f_df, date_col="Datetime", which=None)
                
            else:
                f_df = get_standAlone_dates(df=f_df, date_col="Datetime", which=extract_period)
                
            return f_df
        
        else:
            return f_df


# Summary table functions ----------------------------------------------------------------------------------------
def summary_table(df, gp_vars, num_var, prop_sumy=None):
    """
    parameters
    ----------
    df        [DataFrame]
    gp_vars   [list, str, categorical] A list or a variable to group by.
    num_var   [int64, float64] A variable to summarise.
    prop_sumy [str] Whether to add a proportion summary any of ['median', 'sum', 'mean'].
    
    return
    ------
    summary pandas DataFrame.
    """
    aggregates = ["min", "mean", "median", "max", "sum"]
    
    f_tbl = df.groupby(gp_vars)[num_var].agg(aggregates).reset_index()
    if prop_sumy is not None:
        if prop_sumy not in aggregates:
            raise ValueError(f"argument `prop_sumy` must be any of {aggregates}")
        
        f_tbl[prop_sumy+"_prop"] = round(f_tbl[prop_sumy] / f_tbl[prop_sumy].sum()*100, 2)
    
    return f_tbl



def vars_longer_summary(df, select_vars, var_name, value_name, replace_rec = None, summarise = True): 
    """
    parameter
    ---------
    df          [DataFrame]
    select_vars [list, str, int64, float64] a list of 3 variables from the data, the first var from the list will be used 
                as the `id_vars` and the remaining 2 will be used as the `value_vars` passed to the `melt` function.
    var_name    [str] The name of the melted variable column.
    value_name  [str] The name of the melted value column.
    replace_rec [list, str] A list of two values to replace the default names.
    summarise   [bool] True to summarise the data.
    
    return
    ------
    summary pandas DataFrame.
    """
    f_tbl = df[select_vars].melt(id_vars    = select_vars[0], 
                                 value_vars = select_vars[1:], 
                                 var_name   = var_name, 
                                 value_name = value_name)
    
    if replace_rec is not None:
        f_tbl[var_name] = f_tbl[var_name].replace({select_vars[1]: replace_rec[0], select_vars[2]: replace_rec[1]})
    
    if summarise:
        f_tbl = summary_table(f_tbl, gp_vars = [select_vars[0], var_name], num_var = value_name)
    
    return f_tbl



def sumy_plt_df(df, prop_sumy):
    """
    parameter
    ---------
    df        [DataFrame].
    prop_sumy [str] Function for the proportion summary, any of ['median', 'sum', 'mean'].
    
    return
    ------
    A summary pandas dataframe with an aggregate proportion.
    """
    var_name = df.columns.tolist()[1]
    unique_obs = df[var_name].unique().tolist()
    
    def get_gp_prop(df, v_name, obs, sumy_fun):
        f_df = df.query(f"{v_name} == '{obs}'")
        f_df[sumy_fun+"_prop"] = round(f_df[sumy_fun] / f_df[sumy_fun].sum()*100, 2)
        return f_df

    g_tbl_1 = get_gp_prop(df, var_name, unique_obs[0], prop_sumy)
    g_tbl_2 = get_gp_prop(df, var_name, unique_obs[1], prop_sumy)
    f_tbl = concat([g_tbl_1, g_tbl_2])
    
    return f_tbl




# Ploting functions ----------------------------------------------------------------------------------------------
# matplotlib formmat
# def mp_comma_format(plot, axis):
#     """
#     plot  matplotlib object.
#     axis [string] axis to add comma either 'x', 'y', or 'both'
#     """
#     if axis == "x":
#         plot.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
#     elif axis == "y":
#         plot.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
#     elif axis == "both":
#         plot.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
#         plot.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
#     else:
#         raise ValueError("argument `axis` must be 'x', 'y' or 'both'")
    

    
# ggplot format
def gg_comma_format(axis = "y", deci = 0, prefix = '', surfix = '', **kwags):
    """
    parameter
    ---------
    axis [str] The axis to format the text labels any of 'x', 'y' or  'both'.
    deci [int] The number of decimal places.
    prefix, [str] value to add before and after the text label, must use a sigle quote ''.
    surfix  
    
    return
    ------
    plotnine.scales.scale_xy.scale_y_continuous object
    """
    if axis == "y":
        add = scale_y_continuous(labels = lambda l: [f"{prefix}{{:,.{deci}f}}{surfix}".format(v) for v in l], **kwags) 
    elif axis == "x":
        add = scale_x_continuous(labels = lambda l: [f"{prefix}{{:,.{deci}f}}{surfix}".format(v) for v in l], **kwags) 
    elif axis == "both":
        add = [scale_x_continuous(labels = lambda l: [f"{prefix}{{:,.{deci}f}}{surfix}".format(v) for v in l], **kwags),
               scale_y_continuous(labels = lambda l: [f"{prefix}{{:,.{deci}f}}{surfix}".format(v) for v in l], **kwags) ]
    return add



# creating order text for count plots.
def create_ordered_text(data, cul, f_sort = True, rev = False, axis = "x", **kwargs):
    """
    Parameter
    ---------
    data:  [data.frame] data to use in creating the ordered list.
    cul:   [column] The columns from the data to order.
    f_sort:[bool] True to sort or False not to sort.
    rev:   [bool] True to reverse the list, False not to.
    axis:  [string] The axis to apply the ordered text, 'x'-> x_axis, 'y'-> y_axis.
    kwargs: other argument passed to `scale_x/y_discrete`
    Return
    ------
    if axis = 'x', <plotnine.scales.scale_xy.scale_x_discrete at [env]>
    if axis = 'y', <plotnine.scales.scale_xy.scale_y_discrete at [env]>
    """
    from pandas import value_counts, DataFrame
    from plotnine import scale_x_discrete, scale_y_discrete
    
    if cul not in data.columns.tolist():
        raise Exception(f"variable {cul} is not in the data")
        
    if rev:
        ord_txt = data[cul].value_counts(sort = f_sort).index.tolist()[::-1]
    else:
        ord_txt = data[cul].value_counts(sort = f_sort).index.tolist()
    
    if axis == "x":
        return(scale_x_discrete(limits = ord_txt, **kwargs))
    elif axis == "y":
        return(scale_y_discrete(limits = ord_txt, **kwargs))
    else:
        raise Exception("`axis` can only be either 'x' or 'y'")
        
        

        
def rec_count(df, var, color = None, title = None, x_lab = None, rename_axis_text = None, typ = "plt"):
    """
    parameter
    ---------
    df [DataFrame]
    var [cat, str, object, discrete_floats/int] A variable from the data.
    color [str] color for the bars.
    title [str] title of the plot.
    x_lab [str] x axis label.
    rename_axis_text [list, str] New names for the x axis text.
    typ [str] The output to return 'plt' for plot 'tbl' for a summary table.
    
    return
    ------
    if typ = 'plt' a ggplot object, if 'tbl' then a summary pandas table.
    """
    f_tbl = df[var].value_counts().reset_index(name = "count").rename(columns = {"index": var})
    
    var_l = str.replace(var, "_", " ").title()
    x_lab = var_l if x_lab is None else x_lab
    color = "#7B68EE" if color is None else color
    
    f_plt = (
        ggplot(df, aes(f"factor({var})")) +
        geom_bar(fill = color) +
        geom_text(aes(label = after_stat("prop*100"), group = 1),                                            
                  stat = "count", 
                  nudge_y = 0.125, 
                  va = "bottom",
                  format_string = "{:.2f}%") +
        labs(x = x_lab, y = "Count", title = f"Counts By {var_l}" if title is None else f"Counts By {title}") +
        gg_comma_format("y") +
        create_ordered_text(df, var, axis = "x") +
        theme_light() +
        theme(figure_size= (9, 5))
    )
    
    if rename_axis_text is not None:
        f_plt = f_plt + create_ordered_text(df, var, axis = "x", labels = rename_axis_text)
    
    if typ == "plt":
        return f_plt
    elif typ == "tbl":
        return f_tbl
    else:
        raise ValueError(f"argument `typ` must be 'plt' or 'tbl' and not {typ}")
        

        
def air_quality_count(df, var, color=None):
    color = "#7B68EE" if color is None else color
    sensor = "".join([s for s in var if str.isdigit(s)])
    return (
        ggplot(df, aes(x = var)) +
        geom_bar(fill = color) +
        geom_text(aes(label = after_stat("prop*100"), group = 1),                                            
                  stat = "count", 
                  nudge_y = 0.125, 
                  va = "bottom",
                  format_string = "{:.2f}%") +
        gg_comma_format("y") +
        labs(x = f"Sensor {sensor} AQI", y = "", title = f"Counts Of Records By Sensor {sensor} AQI") +
        theme_light() +
        theme(figure_size=(9, 5))
        )
    
    
# histogram plot 
def histPlot(df, var, bins = None, hue = None, color = None, hue_color = None, zoom = None, title = None, axis_text_suffix = ''):
    """
    parameter
    ---------
    df    [DataFrame]
    var   [int64, float64] A variable from the data `df` to plot the distribution.
    hue   [object, string, category] A variable to from the data `df` to dodge the plot by.
    color [str] color of the bars.
    hue_color [list, dict] A list of colors or a dict of named colors.
    zoom  [list, tupel, int] two numbers to slice the plot by.
    title [str] a name for the plot.
    
    return
    ------
    a ggplot object
    """
    
    var_l = str.replace(var, "_", " ").title()
    color = "#7B68EE" if color is None else color
    title = var_l if title is None else title
    bins = 30 if bins is None else bins
    # plot ------------------------------------------------------------------------------------------
    f_plt = ggplot(df, aes(x = var)) 
    
    if hue is None:
        f_plt = f_plt + geom_histogram(bins = bins, color = "white", fill = color) 
    else:
        f_plt = (f_plt + 
                 geom_histogram(aes(fill = hue), bins = 100, color = "white") +
                 scale_fill_manual(name = var_l, values = hue_color)  
                ) 
        
    f_plt = (f_plt +
             labs(x = var_l, y = "Count", title = f"Distribution of {title}") +
             gg_comma_format(axis = "x", surfix = axis_text_suffix) +
             scale_y_continuous(labels = lambda l: ["{:,.0f}".format(v) for v in l]) +
             theme_light() +
             theme(figure_size= (8, 4))  
            )
    
    if zoom is not None:
        f_plt = f_plt + coord_cartesian(xlim = zoom) 
        
    return f_plt



# Box Plot
def boxPlot(df, num_var, cat_var = None, zoom = None, color=None, title=None, rename_axis_text=None, axis_text_suffix=''):
    """
     parameter
     ---------
     df        [DataFrame]
     num_var   [int64, float64] A variable from the data `df` to plot the distribution.
     cat_var   [str, object, categorical] for Hue plot, a variable from the data `df` to plot the distribution.
     zoom      [list, tupel, int] two numbers to slice the plot by.
     color     [str] a single color for plots without hue, or a list of color with length of the number of unique values in `cat_var`.
     title     [str] a single string for plot without hue or a list of 2 string for plot with hue.
     rename_axis_text [list, str] New names for the x axis text.
     
     return
     ------
     a ggplot object.
    """
    num_var_l = str.replace(num_var, "_", " ").title()
    
    if cat_var is None:
        color = "#788BFF" if color is None else color
        
        f_plt = (ggplot(df, aes(x = 1, y = num_var)) +
                 geom_boxplot(fill = color) +
                 labs(y = num_var_l, title = f"Distribution Of {num_var_l}" if title is None else f"Distribution Of {title}") +
                 gg_comma_format(axis = "y", surfix = axis_text_suffix) +
                 theme_light() +
                 theme(figure_size = (5, 3), 
                      axis_text_x = element_blank(), 
                      axis_title_x = element_blank(), 
                      axis_ticks_major_x = element_blank())
            )
        if zoom is not None:
            f_plt = f_plt + coord_cartesian(ylim = zoom)
            
    else:
        cat_var_l = str.replace(cat_var, "_", " ").title()
        
        f_plt = (ggplot(df, aes(x = cat_var, y = num_var, fill = cat_var)) +
                 geom_boxplot(show_legend = False) +
                 gg_comma_format(axis = "y", surfix = axis_text_suffix) +
                 labs(x = cat_var_l, y = num_var_l, 
                      title = f"Distribution Of {num_var_l} By {cat_var_l}" if title is None else f"Distribution Of {title[0]} By {title[1]}") +
                 theme_light() +
                 theme(figure_size = (5, 4))
        )
        if color is not None:
            f_plt = f_plt + scale_fill_manual(values = color)
        if rename_axis_text is not None:
            f_plt = f_plt + scale_x_discrete(labels = rename_axis_text) 
        if zoom is not None:
            f_plt = f_plt + coord_cartesian(ylim = zoom)
            
    return f_plt



vars_pltTitle = {"faulty": "Device Status", 
                 "Sensor1_PM2.5": "Sensor 1 PM",
                 "Sensor2_PM2.5": "Sensor 2 PM",
                 "S1_AQI": "Sensor 1 Air Quality Index",
                 "S2_AQI": "Sensor 2 Air Quality Index"}


def facet_summary_plot(df, cat_var, num_var, fill, 
                       scales="free", sb_aj=0.25, n_col=2, colors=None, title='', angle=8, plt_size=None):
    """
    df       [DataFrame]
    cat_var  [str, category] A variable from the data.
    num_var  [int64, float64] A variable from the data.
    fill     [object, str, category] A variable from the data to use for the colors.
    scales   [str] facet scales any of ["fixed", "free", "free_y"] passed to `facet_wrap`.
    sb_aj    [float] Subplots adjustment passed to `theme`.
    n_col    [int] Number of facet columns passed to `facet_wrap`.
    title    [str] Plot title.
    colors   [str] Colors of the plot, must be the same length as the unique `fill` variable.
    angle    [float] Plot x axis text direction.
    plt_size [tuple] Size of the plot.
    
    return
    ------
    ggplot object
    """
    
    f_tbl = sumy_plt_df(df, num_var)
    f_tbl[num_var] = f_tbl[num_var].round(2)
    f_tbl[fill] = f_tbl[fill].apply(lambda s: str.replace(s, "_", " "))
    
    colors = ["#A9DEF9", "#F9DEC9"] if colors is None else colors
    plt_size = (13, 4) if plt_size is None else plt_size
    
    return (
        ggplot(f_tbl, aes(x=cat_var, y=num_var, fill=fill)) +
        geom_col(show_legend=False) +
        geom_text(aes(label=num_var), position=position_dodge(width=0.9), va="bottom", size=9, nudge_y=1, color="#B8B8B8") +
        geom_text(aes(label=f"{num_var}_prop"), position=position_dodge(width=0.9), va="top", size=10, format_string="{}%", nudge_y=-1, color="#4A4A4A") +
        facet_wrap(facets=fill, ncol=n_col, scales=scales) +
        scale_fill_manual(values = colors) +
        labs(x="", y="", title=title) +
        theme_light() +
        theme(figure_size=plt_size, 
              strip_background = element_rect(fill="#E5E5E5", color="#E5E5E5"),
              strip_text = element_text(color="#8F8F8F"),
              subplots_adjust={'wspace': sb_aj}, 
              axis_text_x=element_text(angle=angle),
              plot_title=element_text(color="#5C5C5C"))
    )



def facet_summary_plot2(df, cat_var, num_var, fill, facet_by, 
                       scales = "fixed", sb_aj = 0.25, ylab = None, n_col = 2, colors = None, title = '', angle = 0, plt_size = None):
    """
    df       [DataFrame]
    cat_var  [str, category] A variable from the data.
    num_var  [int64, float64] A variable from the data.
    fill     [object, str, category] A variable from the data to use for the colors.
    facet_by [object, str, category] A variable from the data to divide the plot.
    scales   [str] facet scales any of ["fixed", "free", "free_y"] passed to `facet_wrap`.
    sb_aj    [float] Subplots adjustment passed to `theme`.
    n_col    [int] Number of facet columns passed to `facet_wrap`.
    ylab     [str] Y axis label.
    title    [str] Plot title.
    colors   [str] Colors of the plot, must be the same length as the unique `fill` variable.
    angle    [float] Plot x axis text direction.
    plt_size [tuple] Size of the plot.
    
    return
    ------
    ggplot object
    """
    f_tbl = sumy_plt_df(df, num_var)
    plt_size = (9, 4) if plt_size is None else plt_size
    ylab = "" if ylab is None else ylab
    
    colors = ["#75F4F4", "#D17B83"] if colors is None else colors 
    
    f_plt = (
        ggplot(f_tbl, aes(f"reorder({cat_var}, {num_var})", num_var, fill = fill)) + 
        geom_col(show_legend=False) +
        geom_text(aes(label=num_var), position=position_dodge(width=0.9), va="bottom", size=9, nudge_y=1, color="#8F8F8F") +
        geom_text(aes(label=num_var+"_prop"), position=position_dodge(width=0.9), va="top", size=10, format_string="{}%", nudge_y=-1, color="#0A0A0A") +
        facet_wrap(facets=facet_by, ncol=n_col, scales=scales) +           
        gg_comma_format(axis="y") + 
        labs(x="", y=ylab, title=title) +
        scale_fill_manual(values=colors) +
        theme_light() +
        theme(figure_size=plt_size, 
              strip_background=element_rect(fill="#E5E5E5", color="#E5E5E5"),
              strip_text=element_text(color="#8F8F8F"),
              axis_text_x=element_text(angle=angle),
              plot_title=element_text(color="#8F8F8F"))
        )

    if scales in ["free", "free_x","free_y"]:
        f_plt = f_plt + theme(figure_size = plt_size, subplots_adjust = {'wspace': sb_aj}, axis_text_x = element_text(angle = angle))
        
    return f_plt
