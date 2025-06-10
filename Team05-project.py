# Team Project: Sleep Quality and Stress Prediction
# Team Members: Kim Semin, Oh Jinwoo
# Date: 2025.06.11

# This project is pipelined to predict stress levels based on sleep quality and other factors.
# The dataset is read from an Excel file, and the data is preprocessed to prepare for model training.
# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


# 1. Read Excel File
df = pd.read_excel("Dataset.xlsx")

# 2. Variable Name Change
name = {

   ' During the past month, how often have you had trouble sleeping because you: (choose an option that indicate the most accurate reply for the majority of days and nights in the past month) [Cannot get to sleep within 30 minutes]' : 'Cannot get to sleep within 30 minutes',
   ' During the past month, how often have you had trouble sleeping because you: (choose an option that indicate the most accurate reply for the majority of days and nights in the past month) [Wake up in the middle of the night or early morning]' : 'Wake up in the middle of the night or early morning',
   ' During the past month, how often have you had trouble sleeping because you: (choose an option that indicate the most accurate reply for the majority of days and nights in the past month) [Have to get up to use the bathroom]' : 'Have to get up to use the bathroom',
   ' During the past month, how often have you had trouble sleeping because you: (choose an option that indicate the most accurate reply for the majority of days and nights in the past month) [Cough or snore loudly]' : 'Cough or snore loudly',
   ' During the past month, how often have you had trouble sleeping because you: (choose an option that indicate the most accurate reply for the majority of days and nights in the past month) [Feel too cold]' : 'Feel too cold',
   ' During the past month, how often have you had trouble sleeping because you: (choose an option that indicate the most accurate reply for the majority of days and nights in the past month) [Feel too hot]' : 'Feel too hot',
   ' During the past month, how often have you had trouble sleeping because you: (choose an option that indicate the most accurate reply for the majority of days and nights in the past month) [Have bad dreams]'  : 'Have bad dreams',
   ' During the past month, how often have you had trouble sleeping because you: (choose an option that indicate the most accurate reply for the majority of days and nights in the past month) [Have pain]' : 'Have pain',


    'What is your stress level in these given situations [You have to submit an assignment in less than a day]' : 'You have to submit an assignment in less than a day',
    'What is your stress level in these given situations [A week before exams]' : 'A week before exams',
    'What is your stress level in these given situations [Asking for an extra ketchup packet at a restaurant]': 'Asking for an extra ketchup packet at a restaurant',
    'What is your stress level in these given situations [Meeting a new person ]' : 'Meeting a new person',
    'What is your stress level in these given situations [Asking for help]' : 'Asking for help',
    'What is your stress level in these given situations [Confronting someone]' : 'Confronting someone',
    'What is your stress level in these given situations [Doing something without help]' : 'Doing something without help',


    'Gender': 'Gender',
    'Age': 'Age',

    'Have you ever been diagnosed with a mental health condition by a professional (doctor, therapist, etc.)?' : 'Mental health diagnosis Experience',
    'Have you ever received treatment/support for a mental health problem?' : 'Mental health treatment Expereince',

    'When have you usually gone to bed in the past month?' : 'Usual bedtime',
    'How long has it taken you to fall asleep each night in the past month?' : 'Sleep latency',
    'What time have you usually gotten up in the morning in the past month?' : 'Wake-up time',
    'How many hours of actual sleep did you get on an average for the past month? (maybe different from the number of hours spent in bed)' : 'Sleep duration',

}

df_2 = df.rename(columns=name)


# 3. Discard Unnecessary Columns
df_2 = df_2.drop(columns=['Timestamp', 'Your major'])

# drop rows where gender is 'Non-binary'
df_3 = df_2[df_2['Gender'] != 'Non-binary']
df_3 = df_3.reset_index(drop=True)

# 4. Change Data Types

# List for Stress Situation Questions
stress_situation = [
        'You have to submit an assignment in less than a day',
       'A week before exams',
       'Asking for an extra ketchup packet at a restaurant',
       'Meeting a new person', 'Asking for help', 'Confronting someone',
       'Doing something without help'
    
]

# Categorical Order for Stress Situation
stress_level_map = {
    'not stressed': 0,
    'mild': 1,
    'moderate': 2,
    'severe': 3,
    'very severe': 4
}

for col in stress_situation:
    if col in df_3.columns:
        df_3[col] = df_3[col].map(stress_level_map)



stress_cols = ['You have to submit an assignment in less than a day', 'A week before exams', 'Asking for an extra ketchup packet at a restaurant', 'Meeting a new person', 'Asking for help', 'Confronting someone', 'Doing something without help']
df_3['Stress'] = df_3[stress_cols].sum(axis=1)

# 5. Visualization
# Visualization of Stress Score vs. Mental Health Diagnosis History
sns.boxplot(x='Mental health diagnosis Experience', y='Stress', data=df_3)
plt.title("Stress Score vs. Mental Health Diagnosis History")
plt.xlabel("Diangnosis History")
plt.ylabel("Predicted Stress Score")
plt.show()

'''
Yes 그룹은 Stress Score 중앙값이 대략 12.5 ~ 13
No 그룹은 중앙값이 9 정도 → 전체적으로 분산도 큼
Score 15 이상은 Diagnosis 경험자에게서 비교적 많음
'''

# Visualization of Stress Score vs. Mental Health Treatment History
sns.boxplot(x='Mental health treatment Expereince', y='Stress', data=df_3)
plt.title("Stress Score vs. Mental Health Treatment History")
plt.xlabel("Treatment History")
plt.ylabel("Predicted Stress Score")
plt.show()

'''
Yes 그룹은 중앙값이 약 14, 75% 이상이 17 전후
No 그룹은 중앙값이 8~9, 전체적으로 확연히 낮음
치료 경험자들의 stress score 분포가 더 높고 일관됨
'''

# 6. Visualization Checking
'''
박스플롯 분석 결과를 반영하여 Stress level >= 12을 스트레스 위험 구간으로 새롭게 정의하였습니다.
'''

def assign_stress_class(score):
    if score >= 12:
        return 'High'    # 치료 필요했던 그룹에서 다수 분포
    else:
        return 'Low'     # 안정적 구간
df_3['Stress_Level'] = df_3['Stress'].apply(assign_stress_class)

# 7. Data Type Change
sleep_quality = [
    'Cannot get to sleep within 30 minutes',
    'Wake up in the middle of the night or early morning',
    'Have to get up to use the bathroom',
    'Cough or snore loudly',
    'Feel too cold',
    'Feel too hot',
    'Have bad dreams',
    'Have pain'
]

quality_score_map = {
    'Not during the past month': 0,
    'Less than once a week': 1,
    'Once or twice a week': 2,
    'Three or more times a week': 5
}

for col in sleep_quality:
    df_3[col] = df_3[col].map(quality_score_map).fillna(0).astype(int)

df_3['Sleep_Quality_Total'] = df_3[sleep_quality].sum(axis=1)

sleep_latency_map = {
    'under 30 minutes': 0,
    '30 minutes': 1,
    '1 hour': 2,
    '1.5 hours': 3,
    '2 hours': 4,
    'More time than 2 hours': 5 
}
df_3['Sleep_latency_numeric'] = df_3['Sleep latency'].map(sleep_latency_map)

df_3['Usual bedtime'] = df_3['Usual bedtime'].map({
    '9pm-11pm': -2,
    '11pm-1am': 5,
    '1am-3am': 20,
})

df_3['Sleep duration'] = df_3['Sleep duration'].map({
    'less than 4 hours of sleep': 20,
    '4-6 hours': 10,
    '7-8 hours': 3,
    'more than 8 hours': 9
})

# 8. Model Setting
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# 범주형 변수 지정
category_cols = ['Gender', 'Wake-up time']

# 수치형 및 기타 중요 feature 목록
numeric_cols = [
    'Sleep_Quality_Total',
    'Sleep_latency_numeric',
    'Usual bedtime',
    'Sleep duration'
]

# 범주형은 'category' 타입으로 변환
df_3[category_cols] = df_3[category_cols].astype('category')

input_cols = category_cols + numeric_cols
# 최종 feature set 구성
X = df_3[category_cols + numeric_cols]


le = LabelEncoder()
df_3['Stress_Class'] = le.fit_transform(df_3['Stress_Level'])
y = df_3['Stress_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Model Training + Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [-1, 3, 5, 7],
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 400, 800],
}

grid = GridSearchCV(
    estimator=LGBMClassifier(
        class_weight = {
        0: 3,  # High
        1: 2,  # Low
        },
        random_state=42,
        verbose = -1),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=0,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)

best_model = grid.best_estimator_
test_preds = best_model.predict(X_test)
print(classification_report(y_test, test_preds))

# 10. Input Data Preprocessing
def predict_stress_level(input_data, model):
    input_df = pd.DataFrame([input_data])

    # 범주형 매핑
    input_df['Usual bedtime'] = input_df['Usual bedtime'].map({
        '9pm-11pm': -2,
        '11pm-1am': 5,
        '1am-3am': 20,
    })

    input_df['Sleep duration'] = input_df['Sleep duration'].map({
        'less than 4 hours of sleep': 20,
        '4-6 hours': 10,
        '7-8 hours': 3,
        'more than 8 hours': 9
    })

    input_df['Sleep_latency_numeric'] = input_df['Sleep latency'].map({
        'under 30 minutes': 0,
        '30 minutes': 1,
        '1 hour': 2,
        '1.5 hours': 3,
        '2 hours': 4,
        'More time than 2 hours': 5 
    })

    sleep_quality = [
        'Cannot get to sleep within 30 minutes',
        'Wake up in the middle of the night or early morning',
        'Have to get up to use the bathroom',
        'Cough or snore loudly',
        'Feel too cold',
        'Feel too hot',
        'Have bad dreams',
        'Have pain'
    ]

    quality_score_map = {
        'Not during the past month': 0,
        'Less than once a week': 1,
        'Once or twice a week': 2,
        'Three or more times a week': 5
    }

    for col in sleep_quality:
        input_df[col] = input_df[col].map(quality_score_map).fillna(-1).astype(int)

    input_df['Sleep_Quality_Total'] = input_df[sleep_quality].sum(axis=1)

    input_df['Gender'] = input_df['Gender'].astype('category')
    input_df['Wake-up time'] = input_df['Wake-up time'].astype('category')

    X_input = input_df[input_cols]

    predicted_stress_label = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][0] 

    return predicted_stress_label, proba

# 11. Example Data
def stress_recommendation(label):
    if label == 0:
        return "🔴 스트레스 수준이 높습니다. 전문가 상담을 권장합니다. (High stress detected. Professional counseling is recommended.)"
    else:
        return "🟢 스트레스 수준이 안정적입니다. 현재 수면 습관을 유지하세요. (Stress level is stable. Maintain your current sleep habits.)"

# 11. Example Input Data
sample_input_1 = {
    'Gender': 0,  # Female
    'Age': 0,
    'Usual bedtime': '9pm-11pm',
    'Sleep latency': 'under 30 minutes',
    'Wake-up time': '6 -8 am',
    'Sleep duration': '7-8 hours',
    'Cannot get to sleep within 30 minutes': 'Not during the past month',
    'Wake up in the middle of the night or early morning': 'Not during the past month',
    'Have to get up to use the bathroom': 'Less than once a week',
    'Cough or snore loudly': 'Not during the past month',
    'Feel too cold': 'Not during the past month',
    'Feel too hot': 'Not during the past month',
    'Have bad dreams': 'Not during the past month',
    'Have pain': 'Not during the past month',
}


label, proba = predict_stress_level(sample_input_1, best_model)
stress_message = stress_recommendation(label)

print(f"{stress_message} \n(상담 필요 확률: {proba:.1%})")

sample_input_2 = {
    'Gender': 1,  # Male
    'Age': 0,
    'Usual bedtime': '11pm-1am',
    'Sleep latency': '30 minutes',
    'Wake-up time': '8 -10 am',
    'Sleep duration': '4-6 hours',
    'Cannot get to sleep within 30 minutes': 'Once or twice a week',
    'Wake up in the middle of the night or early morning': 'Once or twice a week',
    'Have to get up to use the bathroom': 'Less than once a week',
    'Cough or snore loudly': 'Less than once a week',
    'Feel too cold': 'Not during the past month',
    'Feel too hot': 'Less than once a week',
    'Have bad dreams': 'Once or twice a week',
    'Have pain': 'Not during the past month',
}

label, proba = predict_stress_level(sample_input_2, best_model)
stress_message = stress_recommendation(label)

print(f"{stress_message} \n(상담 필요 확률: {proba:.1%})")

sample_input_3 = {
    'Gender': 1,
    'Age': 0,
    'Usual bedtime': '1am-3am',
    'Sleep latency': '1.5 hours',
    'Wake-up time': '8 -10 am',
    'Sleep duration': 'less than 4 hours of sleep',
    'Cannot get to sleep within 30 minutes': 'Three or more times a week',
    'Wake up in the middle of the night or early morning': 'Three or more times a week',
    'Have to get up to use the bathroom': 'Three or more times a week',
    'Cough or snore loudly': 'Three or more times a week',
    'Feel too cold': 'Three or more times a week',
    'Feel too hot': 'Three or more times a week',
    'Have bad dreams': 'Three or more times a week',
    'Have pain': 'Three or more times a week',
}

label, proba = predict_stress_level(sample_input_3, best_model)
stress_message = stress_recommendation(label)

print(f"{stress_message} \n(상담 필요 확률: {proba:.1%})")

