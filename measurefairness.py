'''
Adapted from: "Certifying and removing disparate impact," Feldman et.al.
                (July 2015)

Goal: measure disparate impact for each of our 4 algorithms
disparate impact: untentional bias which occurs when a selection process has
                  widely different outcomes for different groups, even as it
                  appears to be neutral

TO RUN (as of 1.30.20)
    1: Change X_FEATURE_NAME and X_TGT_VALUE to desired values
    2: run the command: python3 measurefairness.py
'''
import json

X_FEATURE_NAME = 'sex'     # ex./ "race"
X_TGT_VALUE = 'Female'    # ex./ "black"
LOAD_FILENAME = 'simpleBaselineData.json'

def main():
    all_data = get_data()
    #all_data_dict = json.loads(all_data)
    rand_baseline_guesses = all_data['foolish']
    print(type(rand_baseline_guesses[0]))

    all_people = all_data['people']


    mtrx = get_confusion_matrix(all_people,rand_baseline_guesses)
    print(mtrx)
    print("Target Value:", X_TGT_VALUE)
    print("Goal is value <= 1.25, where value =", get_lr_pos(mtrx))
    print("Goal is value > 0.8, where value =", get_lr(mtrx))


def get_data():
    with open(LOAD_FILENAME) as file:
        data_dict = json.load(file)
    return data_dict
    #with open(LOAD_FILENAME) as file:
    #    data_dict = file.readline()
    #return data_dict

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
        elif (c == 0) and (x != X_TGT_VALUE):   # b: if (c=0 and x=1)
            cnfsn_matrix[1] += 1
        elif (c == 1) and (x == X_TGT_VALUE):   # c: if (c=1 and x=0)
            cnfsn_matrix[2] += 1
        elif (c == 1) and (x != X_TGT_VALUE):   # d: if (c=1 and x=1)
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
        return ("ERR: Divide by 0")
    return sens/(1-spec)

# Returns: likelihood ratio (specificity)/(sensitivity)
# A dataset has disparate impact if lr(C,X) <= 0.8
def get_lr(confusion_matrix):
    sens = get_sensitivity(confusion_matrix)
    spec = get_specificity(confusion_matrix)
    if sens==0:
        return ("ERR: Divide by 0")
    return spec/sens



if __name__ == "__main__":
    main()

'''
test_data = [
{'person_id': '1', 'sex': 'Male', 'race': 'White', 'age': '69', 'juv_fel_count': '0', 'juv_misd_count': '0', 'juv_other_count': '0', 'decile_score': '1', 'priors_count': '0',
 'c_charge_degree': '(F3)', 'c_charge_desc': 'Aggravated Assault w/Firearm', 'num_r_cases': '', 'r_charge_degree': '', 'r_charge_desc': '', 'num_vr_cases': '', 'vr_charge_degree': '', 'vr_charge_desc': '', 'is_recid': '0', 'is_violent_recid': '0'},
{'person_id': '2', 'sex': 'Female', 'race': 'Black', 'age': '20', 'juv_fel_count': '0', 'juv_misd_count': '0', 'juv_other_count': '0', 'decile_score': '1', 'priors_count': '0',
 'c_charge_degree': '(F3)', 'c_charge_desc': 'Aggravated Assault w/Firearm', 'num_r_cases': '', 'r_charge_degree': '', 'r_charge_desc': '', 'num_vr_cases': '', 'vr_charge_degree': '', 'vr_charge_desc': '', 'is_recid': '0', 'is_violent_recid': '0'},
{'person_id': '3', 'sex': 'Male', 'race': 'Other', 'age': '45', 'juv_fel_count': '0', 'juv_misd_count': '0', 'juv_other_count': '0', 'decile_score': '1', 'priors_count': '0',
 'c_charge_degree': '(F3)', 'c_charge_desc': 'Aggravated Assault w/Firearm', 'num_r_cases': '', 'r_charge_degree': '', 'r_charge_desc': '', 'num_vr_cases': '', 'vr_charge_degree': '', 'vr_charge_desc': '', 'is_recid': '0', 'is_violent_recid': '0'},
{'person_id': '4', 'sex': 'Male', 'race': 'White', 'age': '18', 'juv_fel_count': '0', 'juv_misd_count': '0', 'juv_other_count': '0', 'decile_score': '1', 'priors_count': '0', 'c_charge_degree': '(F3)',
 'c_charge_desc': 'Aggravated Assault w/Firearm', 'num_r_cases': '', 'r_charge_degree': '', 'r_charge_desc': '', 'num_vr_cases': '', 'vr_charge_degree': '', 'vr_charge_desc': '', 'is_recid': '0', 'is_violent_recid': '0'}]

test_guesses = [0,1,1,0]
'''
