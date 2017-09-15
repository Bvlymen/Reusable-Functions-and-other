#project_specific_functions
from collections import Counter
import pandas as pd
import numpy as np
import re
import operator
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap, Normalize
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
import time


def Create_Regex_Count(Dataframe_Old, Column_Location, Regex, Associated_Count_Location = None, Identifying_Column_Location = None, **re_kwargs):
	"""Looks for Regex in column and creates counter variable for how many times it happened"""
	counter = Counter()
	if (Associated_Count_Location != None) and (Identifying_Column_Location != None):	
		for index, row in enumerate(Dataframe_Old.values):
			if re.search(Regex, row[Column_Location], **re_kwargs):
				counter[row[Identifying_Column_Location]] += row[Associated_Count_Location]
			
	elif (Associated_Count_Location != None) and (Identifying_Column_Location == None):
		for index, row in enumerate(Dataframe_Old.values):
			if re.search(Regex, row[Column_Location], **re_kwargs):
				counter[row[index]] += row[Associated_Count_Location]
			
	elif (Associated_Count_Location == None) and (Identifying_Column_Location != None):
		for index, row in enumerate(Dataframe_Old.values):
			if re.search(Regex, row[Column_Location], **re_kwargs):
				counter[row[Identifying_Column_Location]] += 1
	
	else:
		for index, row in enumerate(Dataframe_Old.values):
			if re.search(Regex, row[Column_Location], **re_kwargs):		
				counter[index] += 1


	count_frame = pd.DataFrame.from_dict(counter, orient = "index")
	count_frame.columns = [Regex + "_count"]
	
	if Identifying_Column_Location != None:
		Dataframe_Old = Dataframe_Old.join(count_frame, on = Dataframe_Old.columns[Identifying_Column_Location], how = "left", lsuffix = "_0", rsuffix="_1")
	else:
		Dataframe_Old = Dataframe_Old.join(count_frame, how = "left", lsuffix = "_0", rsuffix="_1")

	updated_frame = Dataframe_Old.copy()	
	updated_frame.iloc[:,-1].fillna(value=0, inplace = True)

	return updated_frame


def Create_Dummy_From_Count_Above_Threshold(Dataframe_Old, Column, Threshold = 0):
    """Dummy variable for a column based on a threshold value within that column"""
    frame = Dataframe_Old.copy()
    
    frame[str(column + "_dummy")] = Dataframe_Old[Column].map(lambda x: 1 if x> Threshold else 0)

    return frame



                
def Multi_Threshold_Classification_TP_Rate(X, y, Model, Threshold = [None]):
    """ X: Array_like, shape(N_samples, N_Features)
        y: Array_like, shape(N_samples,)
        Threshold: List 0<float<1
        Model: sklearn classification model
    """    
    #Set default threshold as double uplift on randomness
    if Threshold.all() ==None:
        Threshold = [2/np.nunique(y)]
    
    #Initialize dictionaries to hold key(threshold):value(label index/label lists/TP score) pairs
    class_index_dict = {} 
    class_labels_dict = {}
    threshold_TP_score_dict = {}
    
    #Pre-calculate probabilities to speed computation
    probability_predictions = Model.predict_proba(X)
    
    #Calculate the TP score for each threshold value
    for thresh in Threshold:
        class_index_dict[thresh] = [] #Initialize the list to hold indicies for a threshold
        
        #Predict the probabilities to compare to threshold
        for array in probability_predictions:
            #Compare highest (label classified to) probabnility to threshold
            if array.max() > thresh:
                class_index_dict[thresh].append(np.argmax(array))#If certain enough, append the index (classify) of class label.
        #else label the data as unclassified
            else:
                class_index_dict[thresh].append("unclassified")

        class_labels_dict[thresh] = []
        
        for j in class_index_dict[thresh]:
            if j == "unclassified":
                class_labels_dict[thresh].append("unclassified")
            else:
                class_labels_dict[thresh].append(Model.classes_[j])

        score_count=0
        
        for idx,label in enumerate(class_labels_dict[thresh]):
            
            if label == y[idx]:
                score_count +=1
            else:
                pass
        
        classified_boolean_array = np.array(class_labels_dict[thresh])!= "unclassified"
        n_classified = len(y[classified_boolean_array])
        
        threshold_TP_score_dict[thresh] = (score_count/n_classified)

    print(max(threshold_TP_score_dict.items(), key = operator.itemgetter(1)))
     
    return threshold_TP_score_dict    



def Multi_Threshold_Classification_Confusion_Matrix(X, y, Model, Threshold = np.array([None])):
    """ X: Array_like, shape(N_samples, N_Features)
        y: Array_like, shape(N_samples,)
        Threshold: List 0<float<1
        Model: sklearn classification model
    """    
    #Set default threshold as double uplift on randomness
    if Threshold.all() ==None:
        Threshold = [2/len(np.unique(y))]
    
    #Initialize dictionaries to hold key(threshold):value(label index/label lists/TP score) pairs
    class_index_dict = {} 
    class_labels_dict = {}
    threshold_TP_score_dict = {}
    
    #Pre-calculate probabilities to speed computation
    probability_predictions = Model.predict_proba(X)
    
    #Calculate the TP score for each threshold value
    for thresh in Threshold:
        class_index_dict[thresh] = [] #Initialize the list to hold indicies for a threshold
        
        #Predict the probabilities to compare to threshold
        for array in probability_predictions:
            #Compare highest (label classified to) probabnility to threshold
            if array.max() > thresh:
                class_index_dict[thresh].append(np.argmax(array))#If certain enough, append the index (classify) of class label.
        #else label the data as unclassified
            else:
                class_index_dict[thresh].append("unclassified")

        class_labels_dict[thresh] = []
        
        for j in class_index_dict[thresh]:
            if j == "unclassified":
                class_labels_dict[thresh].append("unclassified")
            else:
                class_labels_dict[thresh].append(Model.classes_[j])

        score_count=0
        
        for idx,label in enumerate(class_labels_dict[thresh]):
            
            if label == y[idx]:
                score_count +=1
            else:
                pass
        
        
        INT_length_classifications = len(class_labels_dict[thresh])

        classified_boolean_array = [np.array(class_labels_dict[thresh], dtype= object)!= np.full(INT_length_classifications,"unclassified", dtype= object)]
       
        n_classified = len(y[classified_boolean_array])
        

        threshold_TP_score_dict[thresh] = (score_count/n_classified)

    best_thresh = max(threshold_TP_score_dict.items(), key = operator.itemgetter(1))[0]
    
    print("Old number of samples:", len(X))
    print("New number of samples:", n_classified)

    print("Best True Positive Threshold and Rate: {}".format(max(threshold_TP_score_dict.items(), key = operator.itemgetter(1))))

    #Create_Confusion_Matrix - for each class that we predict, assign an actual class value to

    confusion_matrix_dict = {}
    for label in Model.classes_:
        confusion_matrix_dict[label] = {}
        for prediction in class_labels_dict[best_thresh]:
            confusion_matrix_dict[label][prediction] =0
            
    for idx, prediction in enumerate(class_labels_dict[best_thresh]):
        # if prediction != "unclassified":
        confusion_matrix_dict[y[idx]][prediction] +=1    
        
    confusion_frame = pd.DataFrame(confusion_matrix_dict)
    confusion_frame.rename_axis("Predicted",axis = 0,inplace = True)
    confusion_frame.rename_axis("Actual",axis =1, inplace = True)
    
    return confusion_frame


def Return_Most_Certain_Classification_Data(X, y, Model, Certainty_Thresh = 0, Fit_First = False):
    
    if Fit_First:
        Model = Model.fit(X,y)
    probabilities = Model.predict_proba(X)
    certainty_bool = np.amax(probabilities,axis = 1) > Certainty_Thresh
    
    certain_predictors, certain_response = X[certainty_bool], y[certainty_bool]
    print("Old number of samples:", len(y))
    print("New number of samples:", len(certain_response))
    
    return certain_predictors, certain_response


def Plot_Decision_Boundaries_2D(X1, X2, y , Estimator, Test_Size = 0.3, Random_State = None, Scale = True , Colour_Map = plt.cm.coolwarm, Bright_Colour_Map = plt.cm.afmhot, Alpha_Train = 1, Alpha_Test = 0.6, Certainty_Threshold = None):
    
    def Return_Most_Certain_Classification_Data(X, y, Model, Certainty_Thresh = 0, Fit_First = False):
    
        if Fit_First:
            Model = Model.fit(X,y)
        if hasattr(Model, "predict_proba"):
            probabilities = Model.predict_proba(X)
        elif hasattr(Model, "decision_function"):
            probabilities = Model.decision_function(X)
        certainty_bool = np.amax(probabilities,axis = 1) > Certainty_Thresh
        
        certain_predictors, certain_response = X[certainty_bool], y[certainty_bool]
        print("Old number of samples:", len(y))
        print("New number of samples:", len(certain_response))
        
        return certain_predictors, certain_response

    if Certainty_Threshold != None:
            X_Combined = np.hstack((X1.reshape(-1,1),X2.reshape(-1,1)))
            X, y = Return_Most_Certain_Classification_Data(X_Combined, y, Model = Estimator, Certainty_Thresh = Certainty_Threshold, Fit_First= True)
            X1, X2 = X[:,0], X[:,1]
    
    #Define a class bijection for class colour mapping
    unique_classes, y_bijection = np.unique(y, return_inverse = True)
   
    #Sort the data so colour labels match up with actual labels
    X1 , X2 = X1.reshape((-1,1)), X2.reshape((-1,1))
    y_bijection = y_bijection.reshape((-1,1))
    

    Full_combined = np.hstack((X1,X2, y_bijection))
    Full_combined = Full_combined[Full_combined[:,2].argsort()]

    X1 , X2 = Full_combined[:,0].reshape((-1,1)), Full_combined[:,1].reshape((-1,1))
    y_bijection = Full_combined[:,2].reshape((-1,1))
    
    #Preprocess the data if needed:
    X1, X2 = StandardScaler().fit_transform(X1), StandardScaler().fit_transform(X2)

    delta = 0.02 #Step size in the mesh

    figure = plt.figure(figsize = (12,8))

    x1_min, x1_max = X1.min() -0.5, X1.max() +0.5
    x2_min, x2_max = X2.min() -0.5, X2.max() +0.5

    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, delta), np.arange(x2_min, x2_max, delta))

    
    #Plot the given data (colourmap)

    col_map = Colour_Map
    col_map_bright = Bright_Colour_Map


    #Ready a train test split
    Full_combined = np.hstack((X1, X2, y_bijection))
    

    X_train, X_test, y_train, y_test = train_test_split(Full_combined[:,[0,1]], Full_combined[:,2], test_size = Test_Size, random_state = Random_State)


    ax1 = plt.subplot(1,2,1)

    ax1.set_title("Input Data")
    #Plot Training data
    scat = ax1.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = col_map_bright, edgecolors = 'k', alpha= Alpha_Train)
    #And testing data
    ax1.scatter(X_test[:, 0], X_test[:, 1], c = y_test, cmap = col_map_bright, edgecolors = 'k', alpha =Alpha_Test)

    ax1.set_xlim(xx.min(), xx.max())
    ax1.set_ylim(yy.min(), yy.max())

    ax1.set_xticks(())
    ax1.set_yticks(())


    #Now for the classifier

    ax2 = plt.subplot(1,2,2)

    model = Estimator.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    #Plot the decision boundary. For that, we will assign a colour to each point 
    # in the mesh [x1_min, x1_max]*[x2_min, x2_max]
    
    if hasattr(model, "decision_function"):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])

    elif hasattr(model, "predict_proba"):
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    
    else:
        print("This Estimator doesn't have a decision_function attribute and can't predict probabilities")

    Z= np.argmax(Z, axis = 1)  
    Z_uniques = np.unique(Z)
    
    #Put the result in a colourplot

    Z = Z.reshape(xx.shape)
    
    contour = ax2.contourf(xx, yy, Z, cmap = col_map, alpha=0.9)

    #Plot also the training data
    ax2.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = col_map_bright, edgecolors = 'k', alpha= Alpha_Train)
    #And testing data
    ax2.scatter(X_test[:, 0], X_test[:, 1], c = y_test, cmap = col_map_bright, edgecolors = 'k', alpha = Alpha_Test)

    ax2.set_xlim(xx.min(), xx.max())
    ax2.set_ylim(yy.min(), yy.max())

    ax2.set_xticks(())
    ax2.set_yticks(())

    ax2.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')

    cb1 = plt.colorbar(scat, spacing = "proportional", ax = ax1, ticks = np.arange(len(unique_classes)))
    cb1.ax.set_yticklabels(unique_classes)

    print("Unique Predictions: {}".format(unique_classes[Z_uniques]))
    
    cb2 = plt.colorbar(contour, spacing = "proportional", ax=ax2, ticks = np.arange(len(unique_classes)))
    cb2.ax.set_yticklabels(unique_classes)
    
    #Also print the score of the model
    print(score)

    plt.tight_layout()
    plt.show()

def Remove_Highly_Collinear_Variables(Pandas_Design_Matrix, VIF_Threshold=5, Display_Indicies = True, Return_Scores = False, Verbose = False):
    """
    Removes Highly Collinear variables from a design matrix above a certain covariance threshold

    =========================================
    Design Matrix = Numpy array dim (X,P)
    VIF_Threshold = int/float determining threshold at which variable is removed
    Display_Indicies = Boolean:
                    If True show variable indicies above threshold
    Return_Scores = Boolean:
                    If True then also return the VIF Scores
    Verbose = Boolean:
                    If True then display indicie and iteration in dropping of variables as they happen
    =========================================

    """
    
    VIF_scores = [] #List to which all the high VIF scores will go into
    iteration = 0 #Counter to tell us which iteration we are on
    column_names = [] #List of dropped indicies
    
    PDM_copy = Pandas_Design_Matrix.copy() #Dont want to overwrite old dataframe just in case
    
    
    GO = True #Prepare the while loop
    #Compute the VIF score of each variable if we don't make it to the end of all the columns
    while GO: 
        iteration +=1
        
        #Iterate over current amount of columns
        for i in range(PDM_copy.shape[1]):
            VIF_score = VIF(PDM_copy.values, i)
            
            #IF VIF score above threshold: Drop that variable from the matrix and return a new matrix
            if VIF_score > VIF_Threshold:
                column_name = PDM_copy.columns[i]
                column_names.append(column_name)
                VIF_scores.append(VIF_score)
                PDM_copy = PDM_copy.drop(column_name, axis =1)

                
                #If displaying indicies then say which indicie we dropped
                if Verbose:
                    print("iteration", str(iteration) + ":", "Found high VIF variable with name:", column_name)
                    
                
                
                #Restart looking for high VIF's on new matrix
                STOP = False
                break
            
            #Prepare while loop to stop if the for loop gets to the end
            else:
                STOP = True
            
        #We don't want the while loop to go on forever so we end it because we finished looking for high VIFS.
        if STOP:
            GO = False

    #Say how many variables were dropped
    
    print("\n", "Number of Dropped Variables:", len(VIF_scores), "\n")
    #if wanted, display all the values of the high VIF_scores        
    
    if Return_Scores:
        print("VIF Scores above threshold:", VIF_scores,"\n")
    if Display_Indicies:
        print("Dropped Columns list:", column_names)
    

    return PDM_copy



def Create_Regex_Counts(Dataframe, Regexes, Group_By, Column_Name, Sort=True, Join = False, **Regex_Kwargs):
    
    
    grouped = Dataframe.groupby(by = Group_By, sort = Sort)
    
    if Join:
        DF_new = Dataframe.copy()
    else:
        DF_new = pd.DataFrame(index = list(grouped.groups.keys()))


    def Count_Regex_In_Group(Group_, Regex_, Column_Name_, **Kwargs_):
            
            BOOL_contains = Group_[Column_Name_].str.contains(Regex_, **Kwargs_)
            INT_count = BOOL_contains.sum()
            
            return INT_count

    for idx, regex in enumerate(Regexes,1):
        t0 = time.time() #timeit
        
        S_group_regex_counts = grouped.apply(Count_Regex_In_Group, Regex_= regex, Column_Name_ =Column_Name, **Regex_Kwargs)
        S_group_regex_counts = S_group_regex_counts.rename(str(regex) +"_count")
        
        

        DF_new = DF_new.join(S_group_regex_counts, how ="left")
        
        t1 = time.time() #timeit
        print("Made Regex count",str(idx) +":", str(regex)+ "_count","in:", t1-t0, "seconds")
        

    return DF_new


def Select_Names_From_Value_Counts(Series, Upper, Lower, Max_Names = 20):
    S_value_counts = Series.value_counts()
    
    BOOL_counts_in_range = (S_value_counts<Upper) & (S_value_counts>Lower)
    S_counts_in_range = S_value_counts[BOOL_counts_in_range]
    
    if len(S_counts_in_range) > Max_Names:
        names = S_counts_in_range.index.values[:Max_Names]
    else:
        names = S_counts_in_range.index.values
    
    return names


import numpy as np

def Lift_Score(X, y, Estimator,  Target_Class, Portion_Targeted = 0.1, Train_Test_Split =True, Test_Size=0.3):
        
    if Train_Test_Split:
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =Test_Size)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = Estimator.fit(X_train, y_train)

    #Find which column in predictions to look at for ranking
    class_index = np.where(model.classes_ == Target_Class)[0][0]

    #Obtain certainties over predictions:
    if hasattr(model, "predict_proba"):
        prediction_certainties = model.predict_proba(X_test)

    elif hasattr(model, "decision_function"):
        prediction_certainties = model.decision_function(X_test)

    else:
        print("Model doesn't have attribute: decision_function or predict_proba")

    #Obtain certainties for target class only
    if prediction_certainties.ndim >1:
        prediction_certainty_array = prediction_certainties[:, class_index] 
    else:
        prediction_certainty_array = prediction_certainties

    class_prior = sum(y_test == Target_Class)/len(y_test)
    
    #Get indicies of predictions sorted by certainty of correct classification
    sorted_certainty_indicies = np.argsort(prediction_certainty_array)
    
    #Rank the actual labels by certainty they are of class: "Target_Class"
    sorted_actual_labels = y_test[sorted_certainty_indicies]

    #Work out how many samples we want to target
    number_to_target = np.floor(Portion_Targeted * len(y_test))
    number_to_target = int(number_to_target)

    #Create boolean for who we correctly targeted
    correctly_targeted_bool = (sorted_actual_labels[:number_to_target] == Target_Class)
    
    #How many did we get correct
    number_correctly_classified = np.sum(correctly_targeted_bool)
    
    #Calculate lift
    lift_score = (number_correctly_classified)/(number_to_target*class_prior) #Denominator is how many we would have got correct if random

    return lift_score


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def Plot_Lift_Curve(X, y, Estimator, Target_Class = None, Number_of_Splits = 50, Train_Test_Split = True, Test_Size =0.3, Random_State = None):
    """
    Inputs Preictors and Response and Plots the corresponding lift curve
    i.e the True positive rate given a change in the threshold for targeting
    
    Parameters:

    X = ND array
    y = Numpy 1D array
    Estimator = non-fitted sklearn estimator e.g. KNeighborsClassifier()
    Response_Class = The classification of those samples we are trying to predict
    Train_Test_Split = Boolean  for whether lift curve is plotted on hold_out set or whole dataset

    """
    #Split the Data,if wanted, so that results aren't subject to overfitting
    if Train_Test_Split:
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = Test_Size, random_state = Random_State)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    #Fit the model
    model = Estimator.fit(X_train,y_train)

    #Get predicted classifications
    predictions = model.predict(X_test)
    #Predict scores so that samples can be ranked
    if hasattr(model, "predict_proba"):
        prediction_certainties_NDA = model.predict_proba(X_test)

    elif hasattr(model, "decision_function"):
        prediction_certainties_NDA = model.decision_function(X_test)

    else:
        print("Model doesn't have attribute: decision_function or predict_proba")

    #Find which column in predictions to look at for ranking
    class_index = np.where(model.classes_ == Target_Class)[0][0]

    #Get the target column to rank by certainty
    if prediction_certainties_NDA.ndim >1:
        prediction_certainty_array = prediction_certainties_NDA[:, class_index]
    else:
        prediction_certainty_array = prediction_certainties_NDA


    #Get indicies of predictions sorted by certainty of correct classification
    sorted_certainty_indicies = np.argsort(prediction_certainty_array)
    
    #Rank the actual labels by certainty they are of class: "Target_Class"
    sorted_actual_labels = y_test[sorted_certainty_indicies]
    

    class_prior = sum(y_test == Target_Class)/len(y_test)
    

    def Lift(Sorted_Labels_, Portion_Targeted_, Class_Prior_, Target_Class_):

        #Work out how many samples we want to target
        number_to_target = np.floor(Portion_Targeted_ * len(Sorted_Labels_))
        number_to_target = int(number_to_target)

        #Create boolean for who we correctly targeted
        correctly_targeted_bool = (Sorted_Labels_[:number_to_target] == Target_Class_)
        
        #How many did we get correct
        number_correctly_classified = np.sum(correctly_targeted_bool)
        

        #Calculate lift
        lift_score = (number_correctly_classified)/(number_to_target*Class_Prior_) #Denominator is how many we would have got correct if random

        return lift_score

    #initialise list where lift scores go    
    lift_scores_list = []   
    
    X_percent_targeted = np.arange(0,100, 100/Number_of_Splits)

    for i in X_percent_targeted:
        portion = i+1/(Number_of_Splits)
        lift = Lift(Sorted_Labels_ = sorted_actual_labels, Portion_Targeted_ = portion, Class_Prior_ = class_prior, Target_Class_= Target_Class)

        lift_scores_list.append(lift)
    
    
    plt.plot(X_percent_targeted, lift_scores_list, "r-")
    
    plt.xlabel("percent of of people Targeted")
    plt.title("Lift Curve")
    plt.ylabel("Lift")
    plt.yticks(np.arange(max(lift_scores_list)+2))
    plt.xticks(np.arange(0,100,10))

    plt.show()    


