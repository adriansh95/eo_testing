from scipy.stats import moment

def var_se2(x):
    n = len(x)

    try:
        result = (moment(x, moment=4) - \
                  (n - 3)/(n - 1) * \
                  x.var(ddof=1)**2)/n
    except ZeroDivisionError:
       result = 0 
    
    return result
