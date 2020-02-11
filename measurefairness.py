'''
Adapted from: "Certifying and removing disparate impact," Feldman et.al.
                (July 2015)

Goal: measure disparate impact for each of our 4 algorithms
disparate impact: untentional bias which occurs when a selection process has
                  widely different outcomes for different groups, even as it
                  appears to be neutral

TO RUN
    1: Change X_FEATURE_NAME, X_TGT_VALUE, and X_CMPR_VALUE to desired values
    2: Ensure JSON file (LOAD_FILENAME) is a dictionary in format:
            {'people': [-list of all people-], 'random': [-guesses from random algorithm-], ...}
    2: run the command: python3 measurefairness.py
'''
import json

X_FEATURE_NAME = 'race'     # ex./ "race"
X_TGT_VALUE = 'African-American'    # ex./ "black"
# LOAD_FILENAME = 'simpleBaselineData.json'
LOAD_FILENAME = 'DecisionTreesData.json'
X_CMPR_VALUE = 'Caucasian'     #ex./ "white"


def main():
    all_data = get_data()
    all_keys = [key for key in all_data]
    # first key is all people, followed by all of the algorithm names
    alg_names = all_keys[1:] # remove first key from list

    for alg in alg_names: # test for all algorithms
        print_line()
        print("Algorithm:", alg)
        guesses = all_data[alg]
        all_people = all_data['people']
        display_results(all_people, guesses)
        print_line()


def print_line():
    print("----------------------------")

# Input: a list of people represented as dicts, an int list of guesses (0s and 1s)
# Output: print statements regarding the details of the tests performed
def display_results(people, guesses):
    if (X_FEATURE_NAME == 'age'):
        mtrx = get_age_confusion_matrix(people, guesses)
    else:
        mtrx = get_confusion_matrix(people,guesses)
    print("Confusion Matrix Vals:", mtrx)
    print(X_TGT_VALUE, "total count:", mtrx[0]+mtrx[2])
    print(X_CMPR_VALUE, "total count:", mtrx[1]+mtrx[3])
    print("Feature:", X_FEATURE_NAME)
    print("Testing:", X_TGT_VALUE, "as compared to", X_CMPR_VALUE)

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

    lr = get_lr(matrix)
    if (lr > 0.8):
        print("PASS - value", lr, "> 0.8")
    else:
        print("FAIL - values", lr, "< 0.8")

# loads JSON file containing a dictionary of guess and people data
def get_data():
    with open(LOAD_FILENAME) as file:
        data_dict = json.load(file)
    return data_dict

# Input: a list of dicts represnting people, a list of ints (guesses, 0 or 1)
# Returns: a list of ints [a,b,c,d], each a-d represents a count of people in
#          dataset where: a=(X=0,C=0), b=(X=1,C=0), c=(X=0,C=1),d=(X=1,C=1)
def get_confusion_matrix(all_data,guesses):
    cnfsn_matrix = [0,0,0,0]    # [a,b,c,d]
    for i in range(len(all_data)):
        person = all_data[i]
        c = guesses[i]
        x = person[X_FEATURE_NAME]
        if (c == 0) and (x == X_TGT_VALUE):     # a: if (c=0 and x=0)
            cnfsn_matrix[0] += 1
        elif (c == 0) and (x == X_CMPR_VALUE):   # b: if (c=0 and x=1)
            cnfsn_matrix[1] += 1
        elif (c == 1) and (x == X_TGT_VALUE):   # c: if (c=1 and x=0)
            cnfsn_matrix[2] += 1
        elif (c == 1) and (x == X_CMPR_VALUE):   # d: if (c=1 and x=1)
            cnfsn_matrix[3] += 1

    return cnfsn_matrix

# Identical to "get_confusion_matrix" except it tests for '<=' (ie. age<=34 vs. age>34)
# rather than for equality (ie. race='black' vs. race='white')
def get_age_confusion_matrix(all_data, guesses):
    cnfsn_matrix = [0,0,0,0]    # [a,b,c,d]
    for i in range(len(all_data)):
        person = all_data[i]
        c = guesses[i]
        x = int(person[X_FEATURE_NAME])
        if (c == 0) and (x <= X_TGT_VALUE):     # a: if (c=0 and x=0)
            cnfsn_matrix[0] += 1
        elif (c == 0) and (x > X_CMPR_VALUE):   # b: if (c=0 and x=1)
            cnfsn_matrix[1] += 1
        elif (c == 1) and (x <= X_TGT_VALUE):   # c: if (c=1 and x=0)
            cnfsn_matrix[2] += 1
        elif (c == 1) and (x > X_CMPR_VALUE):   # d: if (c=1 and x=1)
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
