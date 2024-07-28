import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor ,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
import lightgbm as lgb

# Load models
# deforestation_model = load_model('deforestation_model.h5')
# wildfire_model = load_model('wildfire_model.h5')

deforestation_data = pd.read_csv('DeforestationSDG-dataset/goal15.forest_shares.csv')

def load_models():
    deforestation_model = load_model('deforestation_model.h5')
    wildfire_model = load_model('wildfire_model.h5')

    return wildfire_model, deforestation_model 



wildfire_model, deforestation_model = load_models()

# Sidebar
st.sidebar.image('assets/logo1.png', use_column_width=True)
page = st.sidebar.selectbox("Choose a page", ["Why save forests?", "Wildfire Detection", "Deforestation Detection", "Deforestation Analysis"])


# Page 1: Why save forests?
if page == "Why save forests?":
    st.title("The Importance of Forests and the Impact of Deforestation and Wildfires")
    st.write("""
        Forests are vital to the health of our planet. They cover about 31% of the land area on our planet and provide a habitat for over 80% of terrestrial species of animals, plants, and fungi.
        However, deforestation and wildfires pose significant threats to these ecosystems:
        - **Deforestation:** Contributes to 15% of global greenhouse gas emissions and leads to the loss of biodiversity and disruption of water cycles.
        - **Wildfires:** Destroy approximately 4 million hectares of forest every year, releasing large amounts of carbon dioxide and other pollutants into the atmosphere.
    """)

# Page 2: Wildfire Detection
elif page == "Wildfire Detection":
    st.title("Detecting Wildfires Using Satellite Imagery")
    st.write("""
        Wildfires cause immense damage to forests and ecosystems each day. Rapid detection is crucial to mitigating this damage.
    """)
    uploaded_file = st.file_uploader("Upload a satellite image...", type="jpg")
    if uploaded_file is not None:
        test_arr = []
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        test_image = cv2.imdecode(file_bytes, 1)
        # test_image = cv2.imread(uploaded_file)
        test_image = cv2.resize(test_image,(32,32))
        test_image = np.array(test_image)
        test_image = test_image/255
        test_image = test_image.reshape(1,32,32,3)
        test_arr.append(test_image)
        preds = wildfire_model.predict(test_arr)
        nowildfire_prob = preds[0][0]
        wildfire_prob = preds[0][1]

        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        if wildfire_prob> nowildfire_prob:
            st.warning(f"WILDFIRE DETECTED | Probability of Wild Fire: {wildfire_prob:.4f}", icon="🔥")
        else:
            st.success(f"WILDFIRE NOT DETECTED | Probability of Wild Fire: {wildfire_prob:.4f}", icon="🔥")
        

    st.sidebar.write("### Dataset and Model Architecture")
    st.sidebar.write("The model was trained on a dataset of satellite images labeled with wildfire occurrences.")
    st.sidebar.write("Model architecture: Convolutional Neural Network (CNN)")
    st.sidebar.write("The model was trained on a dataset of over 42,000+ satellite images labeled with Wildfire areas.")
    st.sidebar.success("Validation_accuracy: 0.9424 Validation_loss: 0.4649")
    st.sidebar.image("assets/wildfire_model_arch.png", caption="Model Architecture")

# Page 3: Deforestation Detection
elif page == "Deforestation Detection":
    st.title("Detecting Deforestation Using Satellite Imagery")
    st.write("""
        Deforestation causes extensive damage to our environment. Early detection helps in taking prompt actions to prevent further damage.
    """)
    uploaded_file = st.file_uploader("Upload a satellite image...", type="jpg")
    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        print()
        x = x / 255.0
        preds = deforestation_model.predict(x)
        deforestation_prob = preds[0][0]
        pollution_prob = preds[0][1]

        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.warning(f"Deforestation Probability: {deforestation_prob:.4f}", icon="🪓")
        st.warning(f"Pollution Probability: {pollution_prob:.4f}", icon="🎆")

    st.sidebar.write("### Dataset and Model Architecture")
    st.sidebar.write("The model was trained on a dataset of over 40,000 satellite images labeled with deforested areas.")
    st.sidebar.success("Validation_loss: 0.1500, Validation_auc: 0.8841")
    st.sidebar.image("assets\deforestation_model_arch.png", caption="Model Architecture")

# Page 4: Deforestation Analysis
elif page == "Deforestation Analysis":
    st.title("Exploratory Data Analysis on Deforestation Dataset")
    # Load the dataset
    data = pd.read_csv('DeforestationSDG datassset/goal15.forest_shares.csv')
    st.write(data.head())
    # st.write(data.info())
    
    # Most Forest Coverage Countries in 2020 vs 2000 | Deforestation Trend
    top_20 = data.nlargest(20, 'forests_2020')
    top_20 = top_20[::-1]
    plt.figure(figsize=(10, 6))
    top_20.plot(x='iso3c', y=['forests_2000', 'forests_2020'], kind='barh', figsize=(10, 6))
    plt.xlabel('Percentage of Forest Area')
    plt.ylabel('ISO3C')
    plt.title('Top 20 Countries with the Largest Forest Area comparison of Forest Coverage in (2000 vs 2020) - Top 10 Countries Deforestation Trend')
    plt.legend(['2000', '2020'])
    st.pyplot(plt)
    
    plt.figure(figsize=(12, 6))
    for country in top_20['iso3c']:
        plt.plot(['2000', '2020'], [data[data['iso3c'] == country]['forests_2000'].values[0],
                                    data[data['iso3c'] == country]['forests_2020'].values[0]],
                 label=country)
    
    plt.xlabel('Year')
    plt.ylabel('Forest Coverage')
    plt.title('Comparison of Forest Coverage (2000 vs 2020) - Top 10 Countries')
    plt.legend()
    st.pyplot(plt)
    
    # Train and evaluate models
    X = data[['forests_2000', 'forests_2020']]
    y = data['trend']
    
    imputer = SimpleImputer(strategy='mean')
    y = imputer.fit_transform(y.values.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'SVR': SVR(),
        'XGBoost': xgb.XGBRegressor(),
        'LightGBM': lgb.LGBMRegressor(),
        'Gradient Boosting Regressor': GradientBoostingRegressor(),
        'ADA Boost': AdaBoostRegressor(),
        'K Neighbors Regressor': KNeighborsRegressor(),
        'Linear SVR': LinearSVR(),
    }
    
    Name = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR', 'XGBoost', 'LightGBM', 'Gradient Boosting Regressor', 'ADA Boost', 'K Neighbors Regressor', 'Linear SVR']
    accuracy = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        st.write(f"Results for {name}:")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared Score: {r2}")
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(y_test)), y_test, label='Actual Trend')
        plt.plot(np.arange(len(y_test)), y_pred, label='Predicted Trend')
        plt.xlabel('Data Index')
        plt.ylabel('Trend')
        plt.title(f'{name}: Actual Trend vs. Predicted Trend')
        plt.legend()
        st.pyplot(plt)
        accuracy.append(r2)
        st.write()
    
    model_data = pd.DataFrame({"Names": Name, "Accuracies": accuracy})
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Accuracies", y="Names", data=model_data, color="b")
    plt.xlabel("Accuracy")
    plt.ylabel("Model Names")
    plt.title("Model Accuracies")
    st.pyplot(plt)