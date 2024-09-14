#import numpy as np
import pickle
#import pandas as pd
import pandas as pd
#import streamlit as st
import streamlit as st
#import numpy as np
import numpy as np

pickle_in = open("classifier.pkl","rb")
cont_cols_stats=pickle.load(pickle_in)
df=pickle.load(pickle_in)
models=pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_credit_approval(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15):
    A6_0 = 0
    A6_1 = 0
    A6_2 = 0
    A6_3 = 0
    A6_4 = 0
    A6_5 = 0
    A6_6 = 0
    A6_7 = 0
    A6_8 = 0
    A6_9 = 0
    A7_0 = 0
    A7_1 = 0
    A7_2 = 0
    A7_3 = 0
    
    # Box Cox (remove/ minimize skew)
    A2 = (pow(A2 + 0.1, cont_cols_stats[0][1]) - 1) / cont_cols_stats[0][1]
    A3 = (pow(A3 + 0.1, cont_cols_stats[1][1]) - 1) / cont_cols_stats[1][1]
    A8 = (pow(A8 + 0.1, cont_cols_stats[2][1]) - 1) / cont_cols_stats[2][1]
    A11 = (pow(A11 + 0.1, cont_cols_stats[3][1]) - 1) / cont_cols_stats[3][1]
    A14 = (pow(A14 + 0.1, cont_cols_stats[4][1]) - 1) / cont_cols_stats[4][1]
    A15 = (pow(A15 + 0.1, cont_cols_stats[5][1]) - 1) / cont_cols_stats[5][1]

    # Standardization (mean = 0, variance = 1)
    A2 = (A2 - cont_cols_stats[0][2]) / cont_cols_stats[0][3]
    A3 = (A3 - cont_cols_stats[1][2]) / cont_cols_stats[1][3]
    A8 = (A8 + cont_cols_stats[2][2]) / cont_cols_stats[2][3]
    A11 = (A11 + cont_cols_stats[3][2]) / cont_cols_stats[3][3]
    A14 = (A14 - cont_cols_stats[4][2]) / cont_cols_stats[4][3]
    A15 = (A15 - cont_cols_stats[5][2]) / cont_cols_stats[5][3]
    
    column_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
    data = [[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15]]
    df_imputing = pd.DataFrame(data, columns=column_names)
    
    #Imputing
    categorical_cols = ["A1","A4","A5","A6","A7", "A9","A10","A12","A13"]
    cont_cols = ['A2', 'A3', 'A8', 'A11', 'A14','A15']
    
    def euclidean(p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    for col in categorical_cols:
      colcount = df[col].value_counts()
      total_items = sum(colcount)
      colcount = colcount/total_items
    
      imputing_labels = list(colcount[colcount < 0.05].index) #minority is arbitrarily set at 5%
      
      for minority in imputing_labels:
        affected_rows = df.loc[df[col] == minority].index
        
        # Impute for rows where the column has the minority value
        if df_imputing[col][0] == minority:
            k = 3
            all_distances = {}
                
            # Compute distances
            for rownum, comparison in df.iterrows():
                if rownum in affected_rows:
                    continue
                all_distances[rownum] = euclidean(df_imputing[cont_cols], comparison[cont_cols])
                
            # Find top-k neighbors
            topk = sorted(all_distances.keys(), key=lambda x: all_distances[x])[:k]
            topk_labels = [df.loc[top][col] for top in topk]
            final_label = sorted(set(topk_labels), key=topk_labels.count, reverse=True)[0]
            
            # Update the DataFrame
            df_imputing.loc[0, col] = final_label

    
    # Convert DataFrame to NumPy array
    array_imputing = df_imputing.to_numpy()
    
    # Extract values from the array
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15 = array_imputing[0]
    
    if A1 == 'a':
        A1 = 0
    elif A1 == 'b':
        A1 = 1
        
    if A4 == 'u':
        A4 = 0
    elif A4 == 'y':
        A4 = 1
    
    if A5 == 'g':
        A5 = 0
    elif A5 == 'p':
        A5 = 1
        
    if A6 == 'aa':
        A6_0 = 1
    elif A6 == 'c':
        A6_1 == 1
    elif A6 == 'cc':
        A6_2 == 1
    elif A6 == 'ff':
        A6_3 == 1
    elif A6 == 'i':
        A6_4 == 1
    elif A6 == 'k':
        A6_5 == 1
    elif A6 == 'm':
        A6_6 == 1
    elif A6 == 'q':
        A6_7 == 1
    elif A6 == 'w':
        A6_8 == 1
    elif A6 == 'x':
        A6_9 == 1
    
    if A7 == 'bb':
        A7_0 = 1
    elif A7 == 'ff':
        A7_1 = 1
    elif A7 == 'h':
        A7_2 = 1
    elif A7 == 'v':
        A7_3 = 1
    
    if A9 == 'f':
        A9 = 0
    elif A9 == 't':
        A9 = 1
        
    if A10 == 'f':
        A10 = 0
    elif A10 == 't':
        A10 = 1
    
    if A12 == 'f':
        A12 = 0
    elif A12 == 't':
        A12 = 1
    
    if A13 == 'g':
        A13 = 0
    elif A13 == 's':
        A13 = 1
    
    x_test = [[A1,A2,A3,A4,A5,A8,A9,A10,A11,A12,A13,A14,A15,A6_0,A6_1,A6_2,A6_3,A6_4,A6_5,A6_6,A6_7,A6_8,A6_9,A7_0,A7_1,A7_2,A7_3]]
    
    model_names = ["ANN","KNN","SVM"]
    model_prediction = pd.DataFrame()
    for m, mn in zip(models, model_names):
        model_prediction[mn] = m.predict(x_test)
    
    print(model_prediction)
    
    prediction = model_prediction.mode(axis="columns")
    
    predicted_value = prediction.iloc[0, 0]
    
    if predicted_value == 0:
        prediction = '+'
    else:
        prediction = '-'
    
    return prediction



# def main():
#     st.title("CREDIT APPROVAL")
#     html_temp = """
#     <div style="background-color:tomato;padding:10px">
#     <h2 style="color:white;text-align:center;">Streamlit Credit Approval ML App </h2>
#     </div>
#     """
#     st.markdown(html_temp,unsafe_allow_html=True)
    
#     A1 = st.selectbox("A1", ['a', 'b'])
#     A2 = float(st.number_input("A2", value=0.00, step=0.01, format="%.2f"))
#     A3 = float(st.number_input("A3", value=0.000, step=0.001, format="%.3f"))
#     A4 = st.selectbox("A4", ['u', 'y', 'l'])
#     A5 = st.selectbox("A5", ['g', 'p', 'gg'])
#     A6 = st.selectbox("A6", ['c', 'q', 'w', 'i', 'aa', 'ff', 'k', 'cc', 'm', 'x', 'd', 'e', 'j', 'r'])
#     A7 = st.selectbox("A7", ['v', 'h', 'bb', 'ff', 'j', 'z', 'dd', 'n', 'o'])
#     A8 = float(st.number_input("A8", value=0.00, step=0.01, format="%.2f"))
#     A9 = st.selectbox("A9", ['t', 'f'])
#     A10 = st.selectbox("A10", ['f', 't'])
#     A11 = float(st.number_input("A11", value=0, step=1))
#     A12 = st.selectbox("A12", ['f', 't'])
#     A13 = st.selectbox("A13", ['g', 's', 'p'])
#     A14 = float(st.number_input("A14", value=0.0, step=0.1, format="%.1f"))
#     A15 = float(st.number_input("A15", value=0, step=1))
    
#     result=""
    
#     if st.button("Predict"):
#         result = predict_credit_approval(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15)
        
#     st.success('The output is {}'.format(result))

def main():
    # Title of the app
    st.title("Credit Approval Prediction")
    
    # Custom HTML for a stylish header
    html_temp = """
    <div style="background-color:#4CAF50;padding:10px;border-radius:10px">
    <h2 style="color:white;text-align:center;">Streamlit Credit Approval ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Create two columns for better input layout
    col1, col2 = st.columns(2)
    
    # First column inputs
    with col1:
        A1 = st.selectbox("A1", ['a', 'b'])
        A2 = float(st.number_input("A2", value=0.00, step=0.01, format="%.2f", min_value=0.00))
        A3 = float(st.number_input("A3", value=0.000, step=0.001, format="%.3f", min_value=0.000))
        A4 = st.selectbox("A4", ['u', 'y', 'l'])
        A5 = st.selectbox("A5", ['g', 'p', 'gg'])
        A6 = st.selectbox("A6", ['c', 'q', 'w', 'i', 'aa', 'ff', 'k', 'cc', 'm', 'x', 'd', 'e', 'j', 'r'])
        A7 = st.selectbox("A7", ['v', 'h', 'bb', 'ff', 'j', 'z', 'dd', 'n', 'o'])
        A8 = float(st.number_input("A8", value=0.00, step=0.01, format="%.2f", min_value=0.00))

    # Second column inputs
    with col2:
        A9 = st.selectbox("A9", ['t', 'f'])
        A10 = st.selectbox("A10", ['f', 't'])
        A11 = float(st.number_input("A11", value=0, step=1, min_value=0))
        A12 = st.selectbox("A12", ['f', 't'])
        A13 = st.selectbox("A13", ['g', 's', 'p'])
        A14 = float(st.number_input("A14", value=0.0, step=0.1, format="%.1f", min_value=0.0))
        A15 = float(st.number_input("A15", value=0, step=1, min_value=0))
    
    # Adding a nice 'Predict' button
    if st.button("Predict", key="predict_button", use_container_width=True):
        result = predict_credit_approval(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15)
        st.success(f"The prediction result is: {result}")
    else:
        st.info("Fill in the form and click Predict to get the result.")

if __name__=='__main__':
    main()
    
    
    