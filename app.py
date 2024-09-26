import streamlit as st
import inflect
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import string
import plotly.express as px
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

nltk.download('punkt')

punctuations = string.punctuation

# Streamlit app configuration
st.set_page_config(
    page_title="Domestic Building Cost", layout='wide', initial_sidebar_state="auto", page_icon="🏘️"
)

# Load configuration
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize authenticator  
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

# User login
authenticator.login()
name=st.session_state["name"]
authentication_status=st.session_state["authentication_status"]
username=st.session_state["username"] 


# If login is successful, show the main app interface
if authentication_status:
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title(f"Welcome {name}")

    # Remaining code for the app interface
    # Preparing text preprocessing function

    def prep_text(text):
        # Function for preprocessing text
        clean_sents = []  # Append clean sentences
        sent_tokens = sent_tokenize(str(text))
        for sent_token in sent_tokens:
            word_tokens = [str(word_token).strip().lower() for word_token in sent_token.split()]
            word_tokens = [word_token for word_token in word_tokens if word_token not in punctuations]
            clean_sents.append(' '.join(word_tokens))
        joined = ' '.join(clean_sents).strip()
        return joined

    p = inflect.engine()

    # Model name or path to model
    checkpoint = "sadickam/vba-distilbert"

    @st.cache_resource
    def load_model():
        return AutoModelForSequenceClassification.from_pretrained(checkpoint)

    @st.cache_resource
    def load_tokenizer():
        return AutoTokenizer.from_pretrained(checkpoint)

    st.title("🏡 Predictive Analysis for Construction Site Using AI")

    with st.expander("About this app", expanded=False):
        st.write(
            """
            - This app was built by the request of Dhanam Properties in 2023-24.
            - It is intended for predicting the construction cost range of domestic buildings (1 and 2 storeys) at the design/planning stage. The predicted cost range excludes land cost.
            """
        )

    # Project Location and floor area
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
    with b1:
        Suburb = st.text_input("Project Suburb", help="Indian country only. States outside India will return wrong predictions")
    with b2:
        Municipality = st.selectbox(
            "Project Municipality",
            ('Nagar Panchayat', 'Municipal Council', 'Municipal Corporation'),
            help="Some Indian Municipalities may not be listed. Select the closest if not listed"
        )
    with b3:
        Region = st.selectbox("Project Region", ('Metropolitan', 'Rural'))
    with b4:
        SubRegion = st.selectbox(
            "Sub-Region", ('Andaman & Nicobar Islands', 'Lakshadweep', 'Main India', 'East India', 'Northeast India', 'North India', 'South India', 'West India')
        )

    project_detail_1 = f"Site is at {Suburb}, {Municipality}, {Region}, {SubRegion}. "

    st.markdown("")  # Provide space between location info and floor area slider

    # Floor area data entry and description
    FloorArea = st.slider("Estimated total floor area", min_value=100, max_value=750, step=1)
    project_detail_3 = f"Total floor area is {p.number_to_words(FloorArea)} square meters."

    # Building materials, solar water heater, and rainwater storage
    with st.sidebar:
        st.markdown('**Specify basic project information**')
        FloorType = st.selectbox(
            "Choose your floor type",
            ('concrete or stone', 'timber', 'other'),
            help="If your floor type is not listed, please choose 'other'"
        )
        FrameType = st.selectbox(
            "Choose your frame type",
            ('timber', 'steel', 'aluminium', 'other'),
            help="If your frame type is not listed, please choose 'other'"
        )
        RoofType = st.selectbox(
            "Choose your roof type",
            ('Tiles', 'Concrete or Slate', 'Fibre Cement', 'Steel', 'Aluminium', 'other'),
            help="If your roof type is not listed, please choose 'other'"
        )
        WallType = st.selectbox(
            "Choose your wall type",
            ('brick double', 'brick veneer', 'concrete or stone', 'fibre cement', 'timber', 'curtain glass', 'steel', 'aluminium', 'other'),
            help="If your wall type is not listed, please choose 'other'"
        )

        bldg_mat = f"Materials include {FloorType} floor, {FrameType} frame, {WallType} external wall, and {RoofType} roof. "
        Storeys = st.selectbox("Number of storeys", ('one storey', 'two storey'))
        
        SolarHotWater = st.selectbox("Solar hot water", ('Yes', 'No'))
        SolarHotWater = "has solar hot water" if SolarHotWater == "Yes" else 'has no solar hot water'

        RainWaterTank = st.selectbox("Project includes rainwater tank", ("Yes", "No"))
        RainWaterTank = 'and has rainwater tank' if RainWaterTank == "Yes" else 'and no rainwater tank'

        project_detail_2 = f"Building is {Storeys} and {SolarHotWater} {RainWaterTank}. {bldg_mat}"

    st.markdown("##### Project Description")
    with st.form(key="my_form"):
        Project_details = st.text_area(
            "The model's prediction is based on the project description below (i.e., input). Select your options in the sidebar and above to modify the project description below",
            project_detail_1 + project_detail_2 + project_detail_3
        )
        submitted = st.form_submit_button(label="💵 Get cost range!")

    def estimate_time_to_build(storeys, floor_area):
        # Example estimation logic for build time
        base_time = 6  # base time in months for one storey
        time_per_sq_meter = 0.05  # additional time per square meter
        additional_storey_time = 4  # additional time for each additional storey in months
        total_time = base_time + (time_per_sq_meter * floor_area)
        if storeys == 'two storey':
            total_time += additional_storey_time
        return round(total_time, 1)

    def estimate_materials_quantity(floor_area, wall_type, frame_type, roof_type):
        # Example estimation logic for materials quantity
        materials = {
            "bricks": floor_area * 60,  # number of bricks per square meter
            "concrete": floor_area * 0.1,  # cubic meters of concrete per square meter
            "timber": floor_area * 0.05,  # cubic meters of timber per square meter
            "steel": floor_area * 0.02,  # tons of steel per square meter
        }
        # Adjust materials based on wall type, frame type, and roof type
        if wall_type in ['brick double', 'brick veneer']:
            materials["bricks"] *= 1.2
        if frame_type == 'steel':
            materials["steel"] *= 1.5
        if roof_type in ['Steel', 'Aluminium']:
            materials["steel"] *= 1.2
        return materials

    if submitted:
        label_list = ['Rs100,000 - Rs300,000', 'Rs300,000 - Rs500,000', 'Rs500,000 - Rs1.2M']

        joined_clean_sents = prep_text(Project_details)

        # Tokenize
        tokenizer = load_tokenizer()
        tokenized_text = tokenizer(joined_clean_sents, return_tensors="pt")

        # Predict
        model = load_model()
        text_logits = model(**tokenized_text).logits
        predictions = torch.softmax(text_logits, dim=1).tolist()[0]
        predictions = [round(a, 3) for a in predictions]

        # Dictionary with label as key and percentage as value
        pred_dict = dict(zip(label_list, predictions))

        # Sort 'pred_dict' by value and index the highest at [0]
        sorted_preds = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)

        # Make dataframe for Plotly bar chart
        df2 = pd.DataFrame({'Cost_Range': [x[0] for x in sorted_preds], 'Likelihood': [x[1] for x in sorted_preds]})

        # Estimate time to build
        build_time = estimate_time_to_build(Storeys, FloorArea)

        # Estimate materials quantity
        materials_quantity = estimate_materials_quantity(FloorArea, WallType, FrameType, RoofType)

        c1, c2, c3 = st.columns([1, 1, 1])

        with c1:
            # Plot graph of predictions
            fig = px.bar(df2, x="Likelihood", y="Cost_Range", orientation="h")
            fig.update_layout(
                template='seaborn',
                font=dict(family="Arial", size=14, color="black"),
                autosize=False,
                width=400,
                height=300,
                xaxis_title="Likelihood of cost range",
                yaxis_title="Cost ranges"
            )
            fig.update_xaxes(tickangle=0, tickfont=dict(family='Arial', color='black', size=14))
            fig.update_yaxes(tickangle=0, tickfont=dict(family='Arial', color='black', size=14))
            fig.update_annotations(font_size=14)  # This changes y-axis, x-axis, and subplot title font sizes

            st.plotly_chart(fig, use_container_width=False)

        with c2:
            st.header("")
            predicted_range = st.metric("Predicted construction cost range", sorted_preds[0][0])
            Prediction_confidence = st.metric("Prediction confidence", f"{round(sorted_preds[0][1] * 100, 1)}%")
            st.metric("Estimated build time (months)", build_time)

        with c3:
            st.header("")
            st.subheader("Estimated Materials Quantity")
            st.write(f"Bricks: {materials_quantity['bricks']} units")
            st.write(f"Concrete: {materials_quantity['concrete']} cubic meters")
            st.write(f"Timber: {materials_quantity['timber']} cubic meters")
            st.write(f"Steel: {materials_quantity['steel']} tons")

            st.success("Great! Cost range, build time, and materials quantity successfully predicted.", icon="✅")

elif authentication_status == False:
    st.error('Username/password is incorrect')

elif authentication_status == None:
    st.warning('Please enter your username and password')
