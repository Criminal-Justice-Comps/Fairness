'''
Adapted from: "Certifying and removing disparate impact," Feldman et.al.
                (July 2015)

Goal: measure disparate impact for each of our 4 algorithms
disparate impact: untentional bias which occurs when a selection process has
                  widely different outcomes for different groups, even as it
                  appears to be neutral

TO RUN
    1: Change X_FEATURE_NAME, X_MAJORITY_CLASS, and X_MINORITY_CLASS to desired values
    2: Ensure JSON file (LOAD_FILENAME) is a dictionary in format:
            {'people': [-list of all people-], 'random': [-guesses from random algorithm-], ...}
    2: run the command: python3 measurefairness.py
'''
import json
import csv

X_FEATURE_NAME = ['sex','race','race', 'race','age','age','age','age']
X_MAJORITY_CLASS = ['Male','Caucasian','Caucasian','Caucasian',30,40,50,60]
LOAD_FILENAME = 'DecisionTreesData.json'
X_MINORITY_CLASS = ['Female', 'African-American','Hispanic','Other',30,40,50,60]

RESULT_HEADERS = ['algorithm','feature','X_MAJORITY_CLASS','X_MINORITY_CLASS','X_MAJORITY_COUNT',
                  'X_MINORITY_COUNT','a','b','c','d','dis_impact_val','has_dis_impact']
ALL_RESULTS = [RESULT_HEADERS]


def main():
    all_data = get_data()
    all_keys = [key for key in all_data]
    # first key is all people, followed by all of the algorithm names
    alg_names = all_keys[1:] # remove first key from list

    for alg in alg_names: # test for all algorithms
        print_line()
        print("TESTING:", alg)
        all_people = all_data['people']
        guesses = []
        if alg == "ANN":
            for person in all_people:
                guesses.append(int(person["prediction"]))
        else:
            temp_guesses = all_data[alg]
            guesses = []
            for i in temp_guesses:
                guesses.append(i[10])
            print(guesses)

        get_results(all_people, guesses, alg)

        filename = 'DisparateImpactReports/'+ alg + 'DisparateImpact.csv'
        make_csv_results_report(filename)

        display_results(all_people, guesses)
        print_line()



def print_line():
    print("----------------------------")

# Input: confusion matrix
# Output: liklihood ratio (float value), 0 or 1 (true or false depending on
#                                           whether disparate impact is present)
def get_disparate_impact(matrix):
    lr = get_lr(matrix)
    if (lr < 0.8):
        return lr, 1
    else:
        return lr, 0

# Input: a list of people represented as dicts, an int list of guesses (0s and 1s)
# Output: adds to ALL_RESULTS
def get_results(people, guesses, alg):
    for i in range(len(X_FEATURE_NAME)):
        if (X_FEATURE_NAME[i] == 'age'):
            mtrx = get_age_confusion_matrix(people, guesses, i)
        else:
            mtrx = get_confusion_matrix(people,guesses, i)

        lr, has_disparate = get_disparate_impact(mtrx)

        # ['algorithm','feature','X_MAJORITY_CLASS','X_MINORITY_CLASS','X_MAJORITY_COUNT',
        #  'X_MINORITY_COUNT','a','b','c','d','dis_impact_val','has_dis_impact']
        results = [alg, X_FEATURE_NAME[i], X_MAJORITY_CLASS[i], X_MINORITY_CLASS[i],
                   mtrx[1]+mtrx[3], mtrx[0]+mtrx[2], mtrx[0], mtrx[1], mtrx[2], mtrx[3], lr, has_disparate]

        ALL_RESULTS.append(results)

# borrowed from split-cat-num.py
def make_filestring(data):
    # creates a string to write to a file based on the passed list
    string = ''
    for person in data:
        for attribute in person:
            string += str(attribute)
            string += ","
        string = string[:-1]
        string += "\n"
    return string

# Input: file name (string)
# Output: creates a csv file
def make_csv_results_report(filename, create_filestring=True):
    # writes a csv file in `filename` based containing `data`
    if create_filestring:
        string = make_filestring(ALL_RESULTS)
    else:
        string = ALL_RESULTS
    with open(filename, 'w') as file:
        file.write(string)


# Input: a list of people represented as dicts, an int list of guesses (0s and 1s)
# Output: print statements regarding the details of the tests performed
def display_results(people, guesses):
    for i in range(len(X_FEATURE_NAME)):
        if (X_FEATURE_NAME[i] == 'age'):
            mtrx = get_age_confusion_matrix(people, guesses, i)
            print("Confusion Matrix Vals:", mtrx)
            print("Under",X_MINORITY_CLASS[i], "total count:", mtrx[0]+mtrx[2])
            print("Over",X_MAJORITY_CLASS[i], "total count:", mtrx[1]+mtrx[3])
            print("Feature:", X_FEATURE_NAME[i])
            print("Testing: under", X_MINORITY_CLASS[i], "as compared to over", X_MAJORITY_CLASS[i])
        else:
            mtrx = get_confusion_matrix(people,guesses, i)
            print("Confusion Matrix Vals:", mtrx)
            print(X_MINORITY_CLASS[i], "total count:", mtrx[0]+mtrx[2])
            print(X_MAJORITY_CLASS[i], "total count:", mtrx[1]+mtrx[3])
            print("Feature:", X_FEATURE_NAME[i])
            print("Testing:", X_MINORITY_CLASS[i], "as compared to", X_MAJORITY_CLASS[i])

        show_pass_fail(mtrx)

# Input: confusion matrix of values [a, b, c, d]
# Output: none, only print statements
# Functionality: determines whether positive likelihood ratio and likelihood
#                ratio of the input matrix show disparate impact. Prints results
def show_pass_fail(matrix):
    lr_pos = get_lr_pos(matrix)
    if (lr_pos <= 1.25):
        print("PASS - value", lr_pos, "<= 1.25")
    else:
        print("FAIL - value", lr_pos, ">= 1.25")
    print_line()
    lr = get_lr(matrix)
    if (lr > 0.8):
        print("PASS - value", lr, "> 0.8")
    else:
        print("FAIL - values", lr, "< 0.8")
    print_line()

# loads JSON file containing a dictionary of guess and people data
def get_data():
    with open(LOAD_FILENAME) as file:
        data_dict = json.load(file)
    return data_dict

# Input: a list of dicts represnting people, a list of ints (guesses, 0 or 1)
# Returns: a list of ints [a,b,c,d], each a-d represents a count of people in
#          dataset where: a=(X=0,C=0), b=(X=1,C=0), c=(X=0,C=1),d=(X=1,C=1)
def get_confusion_matrix(all_data,guesses,index):
    cnfsn_matrix = [0,0,0,0]    # [a,b,c,d]
    for i in range(len(guesses)):
        person = all_data[i]
        c = guesses[i]
        x = person[X_FEATURE_NAME[index]]
        # print(c, x)
        if (c == 0) and (x == X_MINORITY_CLASS[index]):     # a: if (c=0 and x=0)
            cnfsn_matrix[0] += 1
        elif (c == 0) and (x == X_MAJORITY_CLASS[index]):   # b: if (c=0 and x=1)
            cnfsn_matrix[1] += 1
        elif (c == 1) and (x == X_MINORITY_CLASS[index]):   # c: if (c=1 and x=0)
            cnfsn_matrix[2] += 1
        elif (c == 1) and (x == X_MAJORITY_CLASS[index]):   # d: if (c=1 and x=1)
            cnfsn_matrix[3] += 1

    return cnfsn_matrix

# Identical to "get_confusion_matrix" except it tests for '<=' (ie. age<=34 vs. age>34)
# rather than for equality (ie. race='black' vs. race='white')
def get_age_confusion_matrix(all_data, guesses,index):
    cnfsn_matrix = [0,0,0,0]    # [a,b,c,d]
    for i in range(len(all_data)):
        person = all_data[i]
        c = guesses[i]
        x = float(person[X_FEATURE_NAME[index]])
        # TODO : discuss with Javin that these were backwards
        if (c == 0) and (x <= X_MINORITY_CLASS[index]):     # a: if (c=0 and x=0)
            cnfsn_matrix[0] += 1
        elif (c == 0) and (x > X_MAJORITY_CLASS[index]):   # b: if (c=0 and x=1)
            cnfsn_matrix[1] += 1
        elif (c == 1) and (x <= X_MINORITY_CLASS[index]):   # c: if (c=1 and x=0)
            cnfsn_matrix[2] += 1
        elif (c == 1) and (x > X_MAJORITY_CLASS[index]):   # d: if (c=1 and x=1)
            cnfsn_matrix[3] += 1

    return cnfsn_matrix

# Of all people with attribute (X=1), what % are predicted recidivists (C=1)?
# Input: confusion_matrix = [a, b, c, d]
# Output: (# of people where X=1 and C=1)/(# of people where X=1) = (d)/(b+d)
def get_sensitivity(confusion_matrix):
    d = confusion_matrix[3] # (C=1) and (X=1)
    b = confusion_matrix[1] # (C=0) and (X=1)
    if (b+d==0):
        return 0
    return d/(b+d)

# Of all people without attribute (X=0), what % are predicted recidivists (C=1)?
# Input: confusion_matrix = [a, b, c, d]
# Output: (# of people where X=0 and C=1)/(# of people where X=0) = (a)/(a+c)
def get_specificity(confusion_matrix):
    a = confusion_matrix[0] # (C=0) and (X=0)
    c = confusion_matrix[2] # (C=1) and (X=0)
    if (a+c==0):
        return 0
    return a/(a+c)

# Returns: the positivite likelihood ratio
# A dataset has disparate impact if lr_pos(C,X) > 1.25
def get_lr_pos(confusion_matrix):
    sens = get_sensitivity(confusion_matrix)
    spec = get_specificity(confusion_matrix)
    if spec==1:
        return 0 # ("ERR: Divide by 0")
    return sens/(1-spec)

# Returns: likelihood ratio (1-specificity)/(sensitivity)
# A dataset has disparate impact if lr(C,X) <= 0.8
def get_lr(confusion_matrix):
    sens = get_sensitivity(confusion_matrix)
    spec = get_specificity(confusion_matrix)
    if sens==0:
        return 0 # ("ERR: Divide by 0")
    return (1-spec)/sens #should be the inverse of specificity over sensitivity



if __name__ == "__main__":
    main()
