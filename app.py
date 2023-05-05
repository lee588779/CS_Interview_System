from flask import Flask # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource # Api 구현을 위한 Api 객체 import
from flask import request, render_template # request 객체와 render_template 함수를 임포트해
from sklearn.feature_extraction.text import TfidfVectorizer # 코사인 유사도를 측정하기 위한 패키지를 임포트해

app = Flask(__name__) # Flask 객체 선언, 파라미터로 어플리케이션 이름을 넘겨준다.
api = Api(app) # Flask 객체에 Api 객체를 등록한다.

questions = {
    "1": "What is the capital city of South Korea?",
    "2": "Who is the president of the United States?",
    "3": "What is the name of the largest bone in the human body?",
    "4": "How many planets are there in the solar system?",
    "5": "What is the chemical symbol of gold?"
}

# 정답 목록을 딕셔너리 형태로 만들어
answers = {
    "1": "Seoul",
    "2": "Joe Biden",
    "3": "Femur",
    "4": "Eight",
    "5": "Au"
}

# TfidfVectorizer 객체를 생성해
vectorizer = TfidfVectorizer()

@api.route('/quiz') # '/quiz' 경로에 클래스 Resource를 상속받은 Quiz 클래스를 연결해
class Quiz(Resource):
    def get(self): # GET 요청시에 동작하는 메소드
        return render_template('quiz.html', questions=questions) # quiz.html 파일을 렌더링하고 questions 변수를 넘겨줘

    def post(self): # POST 요청시에 동작하는 메소드
        question_id = request.form.get('question_id') # 폼 데이터에서 문제 번호를 가져와
        transcript = request.form.get('transcript') # 폼 데이터에서 음성 인식 결과를 가져와
        answer = answers[question_id] # 정답 목록에서 해당 문제의 정답을 가져와
        vectors = vectorizer.fit_transform([transcript, answer]) # 음성 인식 결과와 정답을 벡터화해
        cosine_similarity = vectors[0].dot(vectors[1].T).toarray()[0][0] # 코사인 유사도를 계산해
        return render_template('result.html', transcript=transcript, answer=answer, cosine_similarity=cosine_similarity) # result.html 파일을 렌더링하고 음성 인식 결과, 정답, 코사인 유사도 변수를 넘겨줘


@api.route('/stt') # '/stt' 경로에 클래스 Resource를 상속받은 STT 클래스를 연결한다.
class STT(Resource):
    def get(self): # GET 요청시에 동작하는 메소드
        return {'message': 'Hello, World!'} # JSON 형태로 응답을 반환한다.

