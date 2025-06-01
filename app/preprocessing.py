import pandas as pd


def get_tst_points(landmark, order = False):
    """Converts the output of hand_landmarks.landmark into a series of x,y,z

    Arguments:
        landmark -- hand_landmarks.landmark

    Keyword Arguments:
        order -- whether to return it sorted by point so (x1, y1, z1) then (x2, y2, z2) (default: {False})

    Returns:
        a Series of 21 * 3 values for the hand_landmark
    """
    x_ls = []
    y_ls = []
    z_ls = []
    
    if order:
        ordered = []
    for p in landmark:
        p = str(p).split('\n')[:-1]
        x = float(p[0].split(' ')[-1])
        y = float(p[1].split(' ')[-1])
        z = float(p[2].split(' ')[-1])
        
        if order:
            ordered.extend([x,y,z])
        else:
            x_ls.append(x)    
            y_ls.append(y)    
            z_ls.append(z)    
    if order:
        result = ordered
    else:
        result = [*x_ls, *y_ls, *z_ls]
    
    return pd.Series(result)

def normalize_hand(x: pd.Series, with_label=True) -> pd.Series:
    """takes a row and returns a normalized row with $(x_i, y_i) = [(x_i - x_0, y_i - y_0)]$ 
    then dividing them by y12 $(x_i, y_i) = (x_i/y_12, y_i/y_12)

    Arguments:
        x           -- row as a pd.series with data of 21 point + label
        with_label  -- Bolean value indicating whether to return the label or not

    Returns:
        hand marks normalized 
    """
    xs = (x[0:-1:3] - x.iloc[0])
    ys = (x[1:-1:3] - x.iloc[1])
    xs = xs/ys.iloc[12]
    ys = ys/ys.iloc[12]
    zs =  x[2:-1:3]
    
    if with_label:
        label = x[-1:-2:-1]
        result = pd.concat([xs, ys, zs, label])
    
    else:
        result = pd.concat([xs, ys, zs])
   
    return result

