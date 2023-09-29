def clean_data(df):
    import pandas as pd
    df = df[['BP','Age','Height','Weight','foot','Growth','Value','Attacking','Skill','Movement','Power','Mentality','Defending',
              'Goalkeeping','Total Stats','Base Stats', 'W/F', 'SM', 'A/W','D/W', 'IR', 'PAC', 'SHO', 'PAS', 'DRI', 'DEF',
              'PHY','OVA']].copy()
    def feet_to_cm(height_str):
        feet, inches = map(int, height_str.replace('"', '').split("'"))
        total_inches = (feet * 12) + inches
        cm = total_inches * 2.54
        return cm
    df['Height'] = df['Height'].apply(feet_to_cm)
    
    df['Weight']=df['Weight'].str.replace("lbs","")
    df['Weight']=pd.to_numeric(df['Weight'], errors='coerce')
    
    def value_calc(value_str):
        value = value_str.replace("€","")
        if 'M' in value:
            value = value.replace("M","")
            value = pd.to_numeric(value, errors='coerce')
            value = value * 1000000
        elif 'K' in value:
            value = value.replace("K","")
            value = pd.to_numeric(value, errors='coerce')        
            value = value * 1000
        else:
            value = pd.to_numeric(value, errors='coerce')        
        return value
    df['Value'] = df['Value'].apply(value_calc)
    
    df['W/F'] = df['W/F'].str.replace(" ★","")
    df['W/F']=pd.to_numeric(df['W/F'], errors='coerce')
    df['SM'] = df['SM'].str.replace("★","")
    df['SM']=pd.to_numeric(df['SM'], errors='coerce')
    df['IR'] = df['IR'].str.replace(" ★","")
    df['IR']=pd.to_numeric(df['IR'], errors='coerce')
    
    df['A/W'] = df['A/W'].fillna('Medium')
    df['D/W'] = df['D/W'].fillna('Medium')
        
    return df

def y_X(df,z):
    
    import pandas as pd
    
    y = df[z]
    categorical = df.select_dtypes('object')
    numerical = df._get_numeric_data().drop([z], axis=1)
    
    from sklearn.preprocessing import MinMaxScaler
    transformer = MinMaxScaler().fit(numerical)
    numerical_scaled = transformer.transform(numerical)
    numerical_scaled = pd.DataFrame(numerical_scaled, columns=numerical.columns)
    
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(drop='first').fit(categorical)
    cols = encoder.get_feature_names_out(input_features=categorical.columns)
    categorical_encode = pd.DataFrame(encoder.transform(categorical).toarray(),columns=cols)
    
    X = pd.concat([numerical_scaled, categorical_encode], axis=1)
    
    return y, X