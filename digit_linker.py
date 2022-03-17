from sklearn.svm import SVC
# sklearn의 svc()를 사용하여 다중분류 진행
import pandas as pd
import os
# 현재 경로로 폴더 설정
currentPath = os.getcwd()
print(currentPath)
os.chdir(currentPath + "\\digit-recognizer\\")

# 데이터 로드
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# y_train은 라벨 값
y_train = train_df["label"]
print(y_train)

# x_train은 라벨을 제외한 값
x_train = train_df.drop(columns="label")
print(x_train)

# 샘플 서브미션
answer = pd.read_csv("sample_submission.csv")
print(answer)

# x_train에는 28*28픽셀 값을, y_train에는 라벨 값을 저장한다.

# 다중 분류기
svm_clf = SVC()
svm_clf.fit(x_train, y_train)

SVC()

# testarr := test_df 예측
testarr = svm_clf.predict(test_df)

# 분류한 결과를 나만의.csv에 예측 값 생성 후 저장
answer["Label"] = testarr
answer
answer.to_csv('linker98.csv', index=False)
