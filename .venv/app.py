import os
import json
import PyPDF2
import google.generativeai as genai
import datetime
import jwt
import logging
import openai as openai
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from functools import wraps

# --- CẤU HÌNH ---
load_dotenv()  # Tải các biến từ file .env
logging.basicConfig(level=logging.INFO)  # Bật logging để dễ dàng xem lỗi

app = Flask(__name__)

# Cấu hình kết nối Database và các biến môi trường
# Lấy chuỗi kết nối từ file .env, nếu không có thì dùng giá trị mặc định
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL',
                                                  'postgresql://postgres:your_strong_password@localhost:5432/postgres')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# SECRET_KEY dùng để ký token JWT, rất quan trọng và phải được giữ bí mật
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_very_secret_key_that_is_long_and_secure')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Khởi tạo các extension
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)  # Dùng để mã hóa mật khẩu
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# --- ĐỊNH NGHĨA MODEL DATABASE ---
# Lớp này ánh xạ tới bảng 'users' trong database
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def __init__(self, email, password):
        self.email = email
        # Khi tạo user mới, mã hóa mật khẩu ngay lập tức bằng bcrypt
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        # Khi đăng nhập, so sánh mật khẩu người dùng nhập với mật khẩu đã mã hóa trong DB
        return bcrypt.check_password_hash(self.password_hash, password)


# --- CÁC HÀM LOGIC ---
def extract_text_from_pdf(pdf_path):
    """Hàm này trích xuất toàn bộ văn bản từ một tệp PDF."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Ghép nối văn bản từ tất cả các trang
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            app.logger.info(f"Đã trích xuất thành công văn bản từ {os.path.basename(pdf_path)}")
            return text
    except Exception as e:
        app.logger.error(f"Lỗi khi đọc tệp PDF: {e}")
        return None


def analyze_cv_with_ai_openai(cv_text):
    """Hàm này gửi văn bản CV đến ChatGPT để phân tích."""
    if not cv_text:
        app.logger.warning("Văn bản CV rỗng, không gửi đến AI.")
        return None

    # Thiết lập client cho mỗi lần gọi hoặc thiết lập một lần ở global
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Prompt vẫn giữ nguyên, nhưng cách gửi đi thì khác
    prompt = f"""
    Bạn là một chuyên gia tuyển dụng nhân sự (HR) giàu kinh nghiệm. Hãy phân tích nội dung của CV sau đây.
    Hãy đánh giá một cách khách quan dựa trên các tiêu chí: kinh nghiệm làm việc, bộ kỹ năng, trình độ học vấn và cách trình bày.

    Vui lòng trả lời dưới dạng một đối tượng JSON hợp lệ, không chứa bất kỳ văn bản nào khác ngoài JSON.
    Đối tượng JSON phải có các trường sau:
    - "score": một số nguyên từ 0 đến 100, đánh giá tổng thể về CV.
    - "strengths": một chuỗi văn bản mô tả những điểm mạnh chính của CV.
    - "weaknesses": một chuỗi văn bản mô tả những điểm yếu hoặc những gì có thể cải thiện.
    - "detected_skills": một mảng (array) các chuỗi (string) liệt kê các kỹ năng chính bạn phát hiện được (ví dụ: ["Python", "Java", "Project Management"]).

    Đây là nội dung CV:
    ---
    {cv_text}
    ---
    """

    try:
        app.logger.info("Đang gửi yêu cầu đến OpenAI (ChatGPT)...")

        # Cách gọi API của OpenAI khác với Google
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Hoặc "gpt-4" nếu bạn có quyền truy cập
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only responds with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" } # Yêu cầu trả về JSON
        )

        # Cách lấy kết quả cũng khác
        json_response_str = response.choices[0].message.content
        app.logger.info("Đã nhận phản hồi từ OpenAI.")
        return json.loads(json_response_str)

    except Exception as e:
        app.logger.error(f"Lỗi khi gọi API của OpenAI hoặc xử lý JSON: {e}")
        return None


def analyze_cv_with_ai(cv_text):
    """Hàm này gửi văn bản CV đến AI và yêu cầu phân tích."""
    if not cv_text:
        app.logger.warning("Văn bản CV rỗng, không gửi đến AI.")
        return None

    # Chọn mô hình AI
    model = genai.GenerativeModel('gemini-1.5-pro-002')

    # Câu lệnh (prompt) hướng dẫn AI phải làm gì. Đây là phần quan trọng nhất!
    prompt = f"""
    Bạn là một chuyên gia tuyển dụng nhân sự (HR) giàu kinh nghiệm. Hãy phân tích nội dung của CV sau đây.
    Hãy đánh giá một cách khách quan dựa trên các tiêu chí: kinh nghiệm làm việc, bộ kỹ năng, trình độ học vấn và cách trình bày.

    Vui lòng trả lời dưới dạng một đối tượng JSON hợp lệ, không chứa bất kỳ văn bản nào khác ngoài JSON.
    Đối tượng JSON phải có các trường sau:
    - "score": một số nguyên từ 0 đến 100, đánh giá tổng thể về CV.
    - "strengths": một chuỗi văn bản mô tả những điểm mạnh chính của CV.
    - "weaknesses": một chuỗi văn bản mô tả những điểm yếu hoặc những gì có thể cải thiện.
    - "detected_skills": một mảng (array) các chuỗi (string) liệt kê các kỹ năng chính bạn phát hiện được (ví dụ: ["Python", "Java", "Project Management"]).

    Đây là nội dung CV:
    ---
    {cv_text}
    ---
    """

    try:
        app.logger.info("Đang gửi yêu cầu đến Google AI...")
        response = model.generate_content(prompt)

        # Dọn dẹp và chuyển đổi chuỗi trả về thành đối tượng JSON
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        app.logger.info("Đã nhận phản hồi từ AI.")
        return json.loads(cleaned_response)
    except Exception as e:
        app.logger.error(f"Lỗi khi gọi API của AI hoặc xử lý JSON: {e}")
        return None


# --- DECORATOR ĐỂ BẢO VỆ API ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(' ')[1]
        if not token:
            return jsonify({'error': 'Thiếu token xác thực'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            # [SỬA LỖI WARNING] Sử dụng db.session.get thay vì User.query.get
            current_user = db.session.get(User, data['user_id'])
            if not current_user:
                return jsonify({'error': 'Không tìm thấy người dùng'}), 401
        except Exception as e:
            return jsonify({'error': 'Token không hợp lệ hoặc đã hết hạn', 'details': str(e)}), 401
        return f(current_user, *args, **kwargs)

    return decorated


# --- API ENDPOINTS ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"error": "Thiếu email hoặc mật khẩu"}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"error": "Email đã tồn tại"}), 409
    new_user = User(email=data['email'], password=data['password'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "Đăng ký thành công!"}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"error": "Thiếu email hoặc mật khẩu"}), 400
    user = User.query.filter_by(email=data['email']).first()
    if user and user.check_password(data['password']):
        # [SỬA LỖI WARNING] Sử dụng datetime.now(datetime.timezone.utc) thay vì utcnow()
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        return jsonify({"token": token}), 200
    return jsonify({"error": "Email hoặc mật khẩu không đúng"}), 401


@app.route('/upload-cv', methods=['POST'])
@token_required
def upload_cv_endpoint(current_user):
    app.logger.info(f"User {current_user.email} is uploading a CV.")
    if 'file' not in request.files:
        return jsonify({"error": "Không có tệp nào được gửi"}), 400
    file = request.files['file']
    if not file.filename or not file.filename.endswith('.pdf'):
        return jsonify({"error": "Chưa chọn tệp hoặc tệp không phải PDF"}), 400

    filename = secure_filename(file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(pdf_path)

    cv_text = extract_text_from_pdf(pdf_path)
    os.remove(pdf_path)
    if not cv_text:
        return jsonify({"error": "Không thể đọc nội dung từ tệp PDF"}), 500

    ai_result = analyze_cv_with_ai(cv_text)
    if not ai_result:
        return jsonify({"error": "Lỗi khi AI phân tích"}), 500

    return jsonify(ai_result), 200


# --- KHỞI ĐỘNG SERVER ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5001)