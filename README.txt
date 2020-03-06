What is Disparate Impact?

Disparate Impact is calculated by the following fraction:
1 - a/(a+c)
---------
  d/(b+d)

 ^ to get the value of this fraction to decrease, either:
    - decrease the numerator (by increasing specificity)
        --> make "c" smaller
    - increase the denominator (by increasing sensitivity)
        --> make "b" smaller

      # a: if (c=0 and x=0)
      # b: if (c=0 and x=1)
      # c: if (c=1 and x=0)
      # d: if (c=1 and x=1)


         | x=0 | x=1
      -----------
    c=0  |  a  |  b
         |     |
    c=1  |  c  |  d

    c = Recid -or- not recid (guess /prediction)
    x = target -or- compare value

    c is our prediction, and x is an unchangeable attribute of our data.
    So the count of people in each column, must stay the same
        ie the sums: a+c and b+d must be unchanged
    To create "more" disparate impact we want to:
        - simultanously increase "a" while descreasing "c"
        - simultanously increase "d" while descreasing "b"
    An example:
        - X_FEATURE_NAME = 'sex'  X_MAJORITY_CLASS = 'Male' X_MINORITY_CLASS = 'Female'
        - Of the men, predict fewer people to recidivate
        - Of the women, predict more people to recidivate
