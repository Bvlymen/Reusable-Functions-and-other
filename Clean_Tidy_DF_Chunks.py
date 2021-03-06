#IN CHUNKS
import numpy as np
import pandas as pd
import re
from math import log10, floor

def Clean_Tidy_DataFrame_Chunks(dataframe_old, Target, Chunksize = 10000, dataframe_new = pd.DataFrame(), how_na = "all", NA_thresh = None, nsf = 5, Date_Series = "All", Comparison_Date = pd.to_datetime("today"), Cat_Series = None, max_categories = 5, Num_Series = None, Messy_Num_Series= None, Number_Pos = 0, Drop_Target = True, Show_NANs = False):

    def Clean_Tidy_DataFrame(dataframe_old , Target, Chunksize = 10000, dataframe_new = pd.DataFrame(), how_na = "all", thresh = None, nsf = 5, Date_Series = "All", Comparison_Date = pd.to_datetime("today"), Cat_Series = None, max_categories = 5, Num_Series = None, Messy_Num_Series= None, Number_Pos = 0):
        import numpy as np
        import pandas as pd
        import re
        from math import log10, floor

        def Drop_Duplicates(dataframe_old_i = dataframe_old):
            return dataframe_old_i.drop_duplicates()
        dataframe_old = Drop_Duplicates(dataframe_old)

        def Drop_NA(dataframe_old_i = dataframe_old , how_i = how_na , thresh_i = NA_thresh):
            return dataframe_old_i.dropna(axis = [0,1], how = how_i, thresh = thresh_i)
        dataframe_old = Drop_NA(dataframe_old)

        def Extract_Target(dataframe_old_i, Target_i):
            target_i = dataframe_old_i[Target_i].copy()
            return target_i
        target = Extract_Target(dataframe_old_i = dataframe_old, Target_i = Target)

        def Extract_Numeric_Columns(dataframe_old_i = dataframe_old, dataframe_new_i = dataframe_new, nsf_i = nsf):
            """Take all float/integer columns and join to new dataframe"""
            dataframe_new_i = dataframe_new_i.reindex(dataframe_old_i.index)
            from math import log10, floor
            for series in dataframe_old_i:
                if (dataframe_old_i.dtypes[series].name == "int64") or (dataframe_old_i.dtypes[series].name == "float64"):
                        dataframe_new_i[series] = dataframe_old_i[series].copy()
                else:
                    continue
            return dataframe_new_i

        dataframe_clean = Extract_Numeric_Columns(dataframe_old_i = dataframe_old)

        def Extract_Categoricals(dataframe_old_i, dataframe_new_i = dataframe_new):
            dataframe_new_i = dataframe_new_i.reindex(dataframe_old_i.index)
            for series in dataframe_old_i:
                if dataframe_old_i[series].dtypes.name == "category":
                    dataframe_new_i[series] = dataframe_old_i[series].copy()
                else:
                    pass
            return dataframe_new_i

        dataframe_clean = dataframe_clean.join(Extract_Categoricals(dataframe_old_i = dataframe_old))

        def Datetime_to_Magnitude(dataframe_old_i = dataframe_old, dataframe_new_i = dataframe_new, Date_Series_i = Date_Series, Comparison_Date_i = Comparison_Date):
            """Takes all columns that are already datatimes or selcted and converts them into a magnitude in a new dataframe"""
            dataframe_new_i = dataframe_new_i.reindex(dataframe_old_i.index)
            if Date_Series_i != None:
                Comparison_Date_i = pd.to_datetime(Comparison_Date_i)
                import re
                pattern = re.compile("datetime([6432]{2})?(\[ns\])?|Timestamp")
                def subtract_dates(date_stamp, comparison_date = None):
                    delta_time = (comparison_date - date_stamp) /np.timedelta64(1, "D")
                    return delta_time
                if Date_Series_i == "All":
                    Date_Series_i = dataframe_old_i.columns.values
                else:
                    pass
                for series in Date_Series_i:
                    if pattern.match(dataframe_old_i.dtypes[series].name):
                         Delta_Time = dataframe_old_i[series].copy().apply(subtract_dates, comparison_date = Comparison_Date_i)
                         dataframe_new_i[series] = Delta_Time
                    else:
                        continue
            else:
                pass
            return dataframe_new_i

        dataframe_clean = dataframe_clean.join(Datetime_to_Magnitude(dataframe_old_i = dataframe_old))

        def Df_Series_to_Categorical(dataframe_old_i, dataframe_new_i = dataframe_new, Cat_Series_i = Cat_Series, max_categories_i = max_categories):
            """Turns all reasonable series into dummies and appends to a a dataframe that's empty or same length as original data"""
            #Initialise new Dataframe to store result to
            #Make sure it has the correct index
            dataframe_new_i = dataframe_new_i.reindex(dataframe_old_i.index)
            if Cat_Series_i != None:
                for series in Cat_Series_i:
                    if dataframe_old_i[series].nunique() < max_categories_i:
                        #Preprocess the series
                        preprocessed_series = dataframe_old_i[series].str.lower()
                        preprocessed_series = preprocessed_series.str.split(pat = " ").str.join("_")
                        #Create Dummies
                        dummies = pd.get_dummies(preprocessed_series , drop_first = False, prefix = str(series + "_dummy"))
                        #Append dummies
                        dataframe_new_i = dataframe_new_i.join(dummies)
                    else:
                        #Don't want a sample that has too many categories such that in distance space points lie far away
                        print("Too many categories in:", series)
                        continue
            else:
                pass
            return dataframe_new_i

        dataframe_clean = dataframe_clean.join(Df_Series_to_Categorical(dataframe_old_i= dataframe_old))

        def Columns_to_Numeric(dataframe_old_i, dataframe_new_i = dataframe_new, Num_Series_i = Num_Series):
            """Extracts all Targeted columns, turns to numeric and appends to dataframe """
            dataframe_new_i = dataframe_new_i.reindex(dataframe_old_i.index)
            if Num_Series_i != None:
                for series in Num_Series_i:
                    numeric_series = pd.to_numeric(dataframe_old_i[series], errors = "coerce")
                    dataframe_new_i = dataframe_new_i.join(numeric_series)
            else:
                pass
            return dataframe_new_i

        dataframe_clean = dataframe_clean.join(Columns_to_Numeric(dataframe_old_i = dataframe_old))

        def Extract_ith_Nums(dataframe_old_i, dataframe_new_i = dataframe_new , Messy_Num_Series_i= Messy_Num_Series , Number_Pos_i = Number_Pos):
            """Takes a pandas.DataFrame.Series and returns the i'th number located within each element as a new column in a new DataFrame"""
            dataframe_new_i = dataframe_new_i.reindex(dataframe_old_i.index)
            if Messy_Num_Series_i != None:
                import re
                def extract_nums_in_series(element):
                    """Extracts the i'th positive/negative integer/float from an element and returns it"""
                    Extracted = re.findall(pattern = "[-\d.]+", string = str(element))
                    return Extracted[Number_Pos_i]
                for series in Messy_Num_Series_i:
                    Extracted = dataframe_old_i[series].apply(extract_nums_in_series).copy()
                    Extracted = Extracted.astype("float", errors = "ignore")
                    dataframe_new_i = dataframe_new_i.join(Extracted)
            else:
                pass
            return dataframe_new_i

        dataframe_clean = dataframe_clean.join(Extract_ith_Nums(dataframe_old_i = dataframe_old))

        if Drop_Target:
            if Target in dataframe_clean.columns:
                del(dataframe_clean[Target])

        return dataframe_clean, target

    from math import floor
    Range_int = int(floor(len(dataframe_old)/Chunksize))

    Explanatory_DF = dataframe_new    #.reindex(index = dataframe_old.index, columns = dataframe_old.columns)
    Target_Series = pd.Series()    #(index = dataframe_old.index, name = Target)

    if Chunksize < len(dataframe_old):
        for i in range(Range_int):

            X, y = Clean_Tidy_DataFrame(dataframe_old = dataframe_old.iloc[Chunksize*(i):Chunksize * (i+1) , :], Target = Target, dataframe_new = dataframe_new, how_na = how_na, thresh = NA_thresh, nsf = nsf, Date_Series = Date_Series, Comparison_Date = Comparison_Date, Cat_Series = Cat_Series, max_categories = max_categories, Num_Series = Num_Series, Messy_Num_Series= Messy_Num_Series, Number_Pos = Number_Pos)

            Explanatory_DF = Explanatory_DF.append(X) #ignore_index = True)
            Target_Series = Target_Series.append(y) # ignore_index = True)

        if len(dataframe_old) % Chunksize !=0:
            X_lastchunk, y_lastchunk = Clean_Tidy_DataFrame(dataframe_old= dataframe_old.iloc[-Chunksize:, :], Target = Target, dataframe_new = dataframe_new, how_na = how_na, thresh = NA_thresh, nsf = nsf, Date_Series = Date_Series, Comparison_Date = Comparison_Date, Cat_Series = Cat_Series, max_categories = max_categories, Num_Series = Num_Series, Messy_Num_Series= Messy_Num_Series, Number_Pos = Number_Pos)

            Explanatory_DF = Explanatory_DF.append(X_lastchunk) #ignore_index = True)
            Target_Series = Target_Series.append(y_lastchunk) #ignore_index = True)

        else:
            pass

    else:
        Explanatory_DF, Target_Series = Clean_Tidy_DataFrame(dataframe_old= dataframe_old, Target = Target, dataframe_new = dataframe_new, how_na = how_na, thresh = NA_thresh, nsf = nsf, Date_Series = Date_Series, Comparison_Date = Comparison_Date, Cat_Series = Cat_Series, max_categories = max_categories, Num_Series = Num_Series, Messy_Num_Series= Messy_Num_Series, Number_Pos = Number_Pos)

    if Show_NANs:
        for column in Explanatory_DF:
            if sum(Explanatory_DF.loc[:,column].isnull()) > 0.5:
                print("Percentage NaN in", column,":", 100*sum(Explanatory_DF.loc[:,column].isnull())/len(Explanatory_DF) ,"%")
        print("Percentage NaN in Target:", 100*sum(Target_Series.isnull())/len(Target_Series), "%" )

    return Explanatory_DF, Target_Series

"""
******************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************
"""



def Clean_Tidy_DataFrame(dataframe_old, Target, dataframe_new = pd.DataFrame(), how_na = "all", thresh = None, nsf = 5, Date_Series = "All", Comparison_Date = pd.to_datetime("today"), Cat_Series = None, max_categories = 5, Num_Series = None, Messy_Num_Series= None, Number_Pos = 0, Drop_Target = True):
    import numpy as np
    import pandas as pd
    import re
    from math import log10, floor

    def Drop_Duplicates(dataframe_old_i = dataframe_old):
        return dataframe_old_i.drop_duplicates()
    dataframe_old = Drop_Duplicates(dataframe_old)

    def Drop_NA(dataframe_old_i = dataframe_old , how_i = how_na , thresh_i = thresh):
        return dataframe_old_i.dropna(axis = [0,1], how = how_i, thresh = thresh_i)
    dataframe_old = Drop_NA(dataframe_old)

    def Extract_Target(dataframe_old_i, Target_i):
        target_i = dataframe_old_i[Target_i].copy()
        return target_i
    target = Extract_Target(dataframe_old_i = dataframe_old, Target_i = Target)

    def Extract_Numeric_Columns(dataframe_old_i = dataframe_old, dataframe_new_i = dataframe_new, nsf_i = nsf):
        """Take all float/integer columns and join to new dataframe"""
        dataframe_new_i = dataframe_new_i.reindex(dataframe_old_i.index)
        from math import log10, floor
        for series in dataframe_old_i:
            if (dataframe_old_i.dtypes[series].name == "int64") or (dataframe_old_i.dtypes[series].name == "float64"):
                    dataframe_new_i[series] = dataframe_old_i[series].copy()
            else:
                continue
        return dataframe_new_i

    dataframe_clean = Extract_Numeric_Columns(dataframe_old_i = dataframe_old)

    def Extract_Categoricals(dataframe_old_i, dataframe_new_i = dataframe_new):
        dataframe_new_i = dataframe_new_i.reindex(dataframe_old_i.index)
        for series in dataframe_old_i:
            if dataframe_old_i[series].dtypes.name == "category":
                dataframe_new_i[series] = dataframe_old_i[series].copy()
            else:
                pass
        return dataframe_new_i

    dataframe_clean = dataframe_clean.join(Extract_Categoricals(dataframe_old_i = dataframe_old))

    def Datetime_to_Magnitude(dataframe_old_i = dataframe_old, dataframe_new_i = dataframe_new, Date_Series_i = Date_Series, Comparison_Date_i = Comparison_Date):
        """Takes all columns that are already datatimes or selcted and converts them into a magnitude in a new dataframe"""
        dataframe_new_i = dataframe_new_i.reindex(dataframe_old_i.index)
        if Date_Series_i != None:
            Comparison_Date_i = pd.to_datetime(Comparison_Date_i)
            import re
            pattern = re.compile("datetime([6432]{2})?(\[ns\])?|Timestamp")
            def subtract_dates(date_stamp, comparison_date = None):
                delta_time = (comparison_date - date_stamp) /np.timedelta64(1, "D")
                return delta_time
            if Date_Series_i == "All":
                Date_Series_i = dataframe_old_i.columns.values
            else:
                pass
            for series in Date_Series_i:
                if pattern.match(dataframe_old_i.dtypes[series].name):
                     Delta_Time = dataframe_old_i[series].copy().apply(subtract_dates, comparison_date = Comparison_Date_i)
                     dataframe_new_i[series] = Delta_Time
                else:
                    continue
        else:
            pass
        return dataframe_new_i

    dataframe_clean = dataframe_clean.join(Datetime_to_Magnitude(dataframe_old_i = dataframe_old))

    def Df_Series_to_Categorical(dataframe_old_i, dataframe_new_i = dataframe_new, Cat_Series_i = Cat_Series, max_categories_i = max_categories):
        """Turns all reasonable series into dummies and appends to a a dataframe that's empty or same length as original data"""
        #Initialise new Dataframe to store result to
        #Make sure it has the correct index
        dataframe_new_i = dataframe_new_i.reindex(dataframe_old_i.index)
        if Cat_Series_i != None:
            for series in Cat_Series_i:
                if dataframe_old_i[series].nunique() < max_categories_i:
                    #Create Dummies
                    dummies = pd.get_dummies(dataframe_old_i[series] , drop_first = True, prefix = str(series + "_dummy"))
                    #Append dummies
                    dataframe_new_i = dataframe_new_i.join(dummies)
                else:
                    #Don't want a sample that has too many categories such that in distance space points lie far away
                    print("Too many categories in:", series)
                    continue
        else:
            pass
        return dataframe_new_i

    dataframe_clean = dataframe_clean.join(Df_Series_to_Categorical(dataframe_old_i= dataframe_old))

    def Columns_to_Numeric(dataframe_old_i, dataframe_new_i = dataframe_new, Num_Series_i = Num_Series):
        """Extracts all Targeted columns, turns to numeric and appends to dataframe """
        dataframe_new_i = dataframe_new_i.reindex(dataframe_old_i.index)
        if Num_Series_i != None:
            for series in Num_Series_i:
                numeric_series = pd.to_numeric(dataframe_old_i[series], errors = "coerce")
                dataframe_new_i = dataframe_new_i.join(numeric_series)
            for series in Num_Series_i:
                print("Percentage NaN in", series, ":", sum(dataframe_new_i[series] == None)/len(dataframe_new_i))
        else:
            pass
        return dataframe_new_i

    dataframe_clean = dataframe_clean.join(Columns_to_Numeric(dataframe_old_i = dataframe_old))

    def Extract_ith_Nums(dataframe_old_i, dataframe_new_i = dataframe_new , Messy_Num_Series_i= Messy_Num_Series , Number_Pos_i = Number_Pos):
        """Takes a pandas.DataFrame.Series and returns the i'th number located within each element as a new column in a new DataFrame"""
        dataframe_new_i = dataframe_new_i.reindex(dataframe_old_i.index)
        if Messy_Num_Series_i != None:
            import re
            def extract_nums_in_series(element):
                """Extracts the i'th positive/negative integer/float from an element and returns it"""
                Extracted = re.findall(pattern = "[-\d.]+", string = str(element))
                return Extracted[Number_Pos_i]
            for series in Messy_Num_Series_i:
                Extracted = dataframe_old_i[series].apply(extract_nums_in_series).copy()
                Extracted = Extracted.astype("float", errors = "ignore")
                dataframe_new_i = dataframe_new_i.join(Extracted)
        else:
            pass
        return dataframe_new_i

    dataframe_clean = dataframe_clean.join(Extract_ith_Nums(dataframe_old_i = dataframe_old))

    return dataframe_clean, target