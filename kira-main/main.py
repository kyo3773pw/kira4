from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
import logging
import mediapipe as mp
import numpy as np
from functools import wraps
import json
import os
import uuid
import cv2
from twilio.rest import Client
from datetime import datetime, timedelta
import random
import secrets
import pytz
#import face_recognition

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc://lindell2_SQLLogin_2:ztg92qokn7@Usuarios1.mssql.somee.com/Usuarios1?driver=ODBC+Driver+17+for+SQL+Server'
#app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc://sa:user123@anyone_else/usuarios?driver=ODBC+Driver+17+for+SQL+Server'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'user123'  # Cambia esto por una clave segura
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Configuración de logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    permissions = db.Column(db.String(200), nullable=True)
    profile_image = db.Column(db.String(255), nullable=True)
    phone_number = db.Column(db.String(15), nullable=True)
    num_preg = db.Column(db.Integer, nullable=True)
    respuesta = db.Column(db.String(255), nullable=True)
    verification_code = db.Column(db.String(6), nullable=True)
    code_expiry = db.Column(db.DateTime, nullable=True)

    def __init__(self, username, password, role, phone_number=None, num_preg=None, 
                 respuesta=None, permissions=None, profile_image=None):
        self.username = username
        self.password = password
        self.role = role
        self.phone_number = phone_number
        self.num_preg = num_preg
        self.respuesta = respuesta
        self.permissions = permissions
        self.profile_image = profile_image


# Configuración de Mediapipe para la malla facial
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

def generate_frames(username):
    blink_count = 0
    blink_detected = False
    clean_image = None

    try:
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while True:
                success, frame = cap.read()
                if not success:
                    logging.warning("No se pudo capturar el frame.")
                    break

                clean_image = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )

                        landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in face_landmarks.landmark]

                        eye_right = landmarks[145], landmarks[159]
                        dist_right_eye = np.linalg.norm(np.array(eye_right[0]) - np.array(eye_right[1]))

                        eye_left = landmarks[374], landmarks[386]
                        dist_left_eye = np.linalg.norm(np.array(eye_left[0]) - np.array(eye_left[1]))

                        if dist_right_eye < 10 and dist_left_eye < 10 and not blink_detected:
                            blink_count += 1
                            blink_detected = True
                        elif dist_right_eye > 10 and dist_left_eye > 10:
                            blink_detected = False

                        cv2.putText(frame, f"Parpadeos: {blink_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        if blink_count >= 2:
                            _, buffer = cv2.imencode('.jpg', clean_image)
                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                            return img_base64

                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        logging.error(f"Error al inicializar FaceMesh: {e}")
        # Aquí podrías decidir qué hacer si falla la inicialización de FaceMesh


# Ruta para transmitir el video en tiempo real
@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(current_user.username), mimetype='multipart/x-mixed-replace; boundary=frame')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_user.role != 'admin':
            flash('No tienes permiso para acceder a esta página.')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def coadmin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_user.role not in ['admin', 'coadmin']:
            flash('No tienes permiso para acceder a esta página.')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def main_index():
    logout_user()
    session.clear()
    """Nueva ruta principal que muestra index2.html"""
    return render_template('index2.html')

@app.route('/index')
@login_required
@admin_required
def index():
    users = User.query.all()
    return render_template('index.html', users=users)

@app.route('/dashboard')
@login_required
def dashboard():
    """Ruta después del login exitoso"""
    if current_user.role == 'admin':
        users = User.query.all()
        return render_template('index.html', users=users)
    return render_template('user_inicio.html')

@app.route('/create_user', methods=['GET', 'POST'])
def create_user():
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            phone_number = request.form.get('phone', '')
            num_preg = request.form.get('num_preg')
            respuesta = request.form.get('respuesta', '').strip()
            mesh_points = request.form.get('profile_image')

            # Validaciones
            if not all([username, password, phone_number, num_preg, respuesta]):
                flash('Todos los campos son requeridos.')
                return redirect(url_for('create_user'))

            if User.query.filter_by(username=username).first():
                flash('El nombre de usuario ya está en uso.')
                return redirect(url_for('create_user'))

            # Generar hash de contraseña
            hashed_password = generate_password_hash(password)
            
            # Procesar malla facial
            mesh_file_db_path = None
            if mesh_points:
                try:
                    mesh_dict = json.loads(mesh_points)
                    faces_dir = os.path.join(app.root_path, 'static', 'faces')
                    os.makedirs(faces_dir, exist_ok=True)
                    mesh_filename = f"{uuid.uuid4().hex}_{username}.json"
                    mesh_file_relative_path = os.path.join('static', 'faces', mesh_filename)
                    mesh_file_path = os.path.join(app.root_path, mesh_file_relative_path)
                    
                    with open(mesh_file_path, 'w', encoding='utf-8') as mesh_file:
                        json.dump(mesh_dict, mesh_file, ensure_ascii=False, indent=4)
                    
                    mesh_file_db_path = mesh_file_relative_path.replace(os.sep, '/')
                    
                except Exception as e:
                    flash(f'Error al procesar la malla facial: {str(e)}')
                    return redirect(url_for('create_user'))

            # Crear nuevo usuario
            try:
                new_user = User(
                    username=username,
                    password=hashed_password,
                    role='user',  # Role por defecto
                    phone_number=phone_number,
                    num_preg=num_preg,
                    respuesta=respuesta,
                    profile_image=mesh_file_db_path
                )
                
                db.session.add(new_user)
                db.session.commit()
                
                flash('Usuario creado correctamente.')
                return redirect(url_for('login'))
                
            except Exception as e:
                db.session.rollback()
                if mesh_file_db_path and os.path.exists(mesh_file_path):
                    try:
                        os.remove(mesh_file_path)
                    except:
                        pass
                flash(f'Error al crear el usuario: {str(e)}')
                return redirect(url_for('create_user'))

        except Exception as e:
            flash(f'Error general: {str(e)}')
            return redirect(url_for('create_user'))

    return render_template('create_user.html')

@app.route('/edit_user/<username>', methods=['GET', 'POST'])
@login_required
@coadmin_required
def edit_user(username):
    user = User.query.filter_by(username=username).first()

    if request.method == 'POST':
        user.username = request.form['username']
        if request.form['password']:
            user.password = generate_password_hash(request.form['password'])
        user.role = request.form['role']

        db.session.commit()
        flash('Usuario actualizado correctamente.')
        return redirect(url_for('index'))

    return render_template('edit_user.html', username=user.username, role=user.role)


@app.route('/delete_user/<username>', methods=['DELETE'])
@login_required
@admin_required
def delete_user_route(username):
    user = User.query.filter_by(username=username).first()
    if user:
        # Eliminar los registros relacionados en verification_log
        VerificationLog.query.filter_by(user_id=user.id).delete()
        
        # Eliminar el usuario
        db.session.delete(user)
        db.session.commit()
        
        flash('Usuario y registros relacionados eliminados correctamente.')
    else:
        flash('El usuario no se encontró.')
    return '', 204  # No Content


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        # Si el usuario ya está autenticado
        if current_user.is_authenticated:
            logout_user()
            session.clear()
        return render_template('login.html')
    

    elif request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')

            if not username or not password:
                return jsonify({'success': False, 'message': 'Usuario y contraseña son requeridos.'})

            user = User.query.filter_by(username=username).first()
            
            if user and check_password_hash(user.password, password):
                # Solo procesa usuarios normales aquí
                if user.role == 'user':
                    if user.profile_image:
                        session['username'] = username
                        return jsonify({
                            'success': True, 
                            'message': 'Credenciales correctas. Proceda a la verificación facial.',
                            'require_facial': True
                        })
                    else:
                        return jsonify({
                            'success': False, 
                            'message': 'El usuario no tiene una malla facial registrada.'
                        })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Por favor use el login de administrador.'
                    })
            else:
                return jsonify({'success': False, 'message': 'Usuario o contraseña incorrectos.'})
        except Exception as e:
            app.logger.error(f"Login error: {str(e)}")
            return jsonify({'success': False, 'message': 'Error en el proceso de inicio de sesión.'})
        
# Configuración de Twilio
TWILIO_ACCOUNT_SID = 'ACac0e9cca0354e194ec3c4666573e5ad9' #no me eja subir mis credenciales asi q falta rellenar
TWILIO_AUTH_TOKEN = 'eec634f708e4f4d42a87a7f981083031' #igual aqui
#TWILIO_WHATSAPP_NUMBER = '+14155238886'  # Formato: 'whatsapp:+14155238886'
TWILIO_WHATSAPP_NUMBER = '+14155238886'  # Formato: 'whatsapp:+14155238886'


client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_whatsapp_verification(phone_number, code):
    """Envía el código de verificación por WhatsApp usando Twilio"""
    try:
        message = client.messages.create(
            from_=f'whatsapp:{TWILIO_WHATSAPP_NUMBER}',  # Agregar prefijo whatsapp:
            body=f"Tu código de verificación es: {code}\nVálido por 30 segundos.",
            #to=f'whatsapp:+{+51993811230}'  # Ya tienes el prefijo whatsapp: aquí
            to=f'whatsapp:+{+51972460207}'
        )
        return True
    except Exception as e:
        app.logger.error(f"Twilio WhatsApp error: {str(e)}")
        return False

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    try:
        if request.method == 'GET':
            return render_template('admin_login.html')
        
        username = request.form.get('username')
        password = request.form.get('password')
        verification_code = request.form.get('verification_code')

        if not username or not password:
            return jsonify({
                'success': False, 
                'message': 'Usuario y contraseña son requeridos.'
            })

        user = User.query.filter_by(username=username).first()
        
        if not user or not check_password_hash(user.password, password):
            return jsonify({
                'success': False, 
                'message': 'Credenciales de administrador incorrectas.'
            })

        if user.role not in ['admin', 'coadmin']:
            return jsonify({
                'success': False,
                'message': 'No tiene permisos de administrador.'
            })

        # Si no hay código de verificación, generar y enviar uno nuevo
        if not verification_code:
            new_code = ''.join(random.choices('0123456789', k=6))
            expiry_time = datetime.now() + timedelta(seconds=30)
            
            user.verification_code = new_code
            user.code_expiry = expiry_time
            db.session.commit()
            
            if send_whatsapp_verification(user.phone_number, new_code):
                return jsonify({
                    'success': True,
                    'requires_2fa': True,
                    'message': 'Se ha enviado un código de verificación a tu WhatsApp.'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Error al enviar el código de verificación.'
                })

        # Verificar el código
        if not user.verification_code or not user.code_expiry:
            return jsonify({
                'success': False,
                'message': 'No hay código de verificación pendiente.'
            })

        if datetime.now() > user.code_expiry:
            user.verification_code = None
            user.code_expiry = None
            db.session.commit()
            
            return jsonify({
                'success': False,
                'message': 'El código de verificación ha expirado.'
            })

        if verification_code != user.verification_code:
            return jsonify({
                'success': False,
                'message': 'Código de verificación incorrecto.'
            })

        # Si todo está correcto, hacer login
        login_user(user)
        login_record = LoginRecord(
                user_id=user.id,
                login_datetime=datetime.now(),
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string
            )
        db.session.add(login_record)
        db.session.commit()
        
        # Limpiar el código usado
        user.verification_code = None
        user.code_expiry = None
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'message': 'Inicio de sesión exitoso como administrador.',
            'redirect': url_for('index')  # Cambiado a la ruta raíz directamente
        })

    except Exception as e:
        app.logger.error(f"Admin login error: {str(e)}")
        return jsonify({
            'success': False, 
            'message': 'Error en el proceso de inicio de sesión de administrador.'
        })
    
# Definición del umbral de similitud
SIMILARITY_THRESHOLD = 0.94

# Definición del modelo VerificationLog (si es necesario)
class VerificationLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    success = db.Column(db.Boolean, nullable=False)
    similarity_score = db.Column(db.Float, nullable=False)
    ip_address = db.Column(db.String(45), nullable=True)

def log_verification_attempt(user_id, success, similarity_score):
    """
    Registra los intentos de verificación facial en la base de datos.
    """
    try:
        # Crear un nuevo registro de intento de verificación (debes tener un modelo VerificationLog)
        log = VerificationLog(
            user_id=user_id,
            timestamp=datetime.utcnow(),
            success=success,
            similarity_score=similarity_score,
            ip_address=request.remote_addr
        )
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        app.logger.error(f"Error al registrar intento de verificación: {str(e)}")

# Definición de la función validate_mesh_structure
def validate_mesh_structure(mesh1, mesh2):
    """
    Valida que ambas mallas faciales tengan la estructura correcta.
    """
    try:
        # Verificar que ambas mallas sean listas
        if not isinstance(mesh1, list) or not isinstance(mesh2, list):
            return False

        # Verificar que ambas mallas tengan la misma longitud
        if len(mesh1) != len(mesh2):
            return False

        # Verificar que cada punto tenga coordenadas x, y, z
        for mesh in [mesh1, mesh2]:
            if not all(
                isinstance(point, dict) and 
                'x' in point and 'y' in point and 'z' in point and
                all(isinstance(coord, (int, float)) for coord in point.values())
                for point in mesh
            ):
                return False

        return True
    except Exception as e:
        app.logger.error(f"Error validando estructura de malla: {str(e)}")
        return False
    
def compare_facial_meshes(mesh1, mesh2):
    """
    Compara dos mallas faciales y retorna un puntaje de similitud.
    """
    try:
        # Convertir las listas de puntos a arrays numpy para cálculos matemáticos
        points1 = np.array([[point['x'], point['y'], point['z']] for point in mesh1])
        points2 = np.array([[point['x'], point['y'], point['z']] for point in mesh2])
        
        # Normalizar los puntos para eliminar diferencias de escala y posición
        points1_norm = normalize_points(points1)
        points2_norm = normalize_points(points2)
        
        # Calcular las distancias entre puntos correspondientes
        distances = np.linalg.norm(points1_norm - points2_norm, axis=1)
        
        # Calcular la similitud (por ejemplo, una distancia promedio inversa)
        similarity_score = 1.0 / (1.0 + np.mean(distances))
        
        return similarity_score
        
    except Exception as e:
        app.logger.error(f"Error en comparación de mallas: {str(e)}")
        return 0.0

def normalize_points(points):
    """
    Normaliza los puntos para hacer la comparación invariante a escala y posición.
    """
    # Centrar los puntos en el origen
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Escalar para que el tamaño de la malla sea uniforme
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    
    return normalized

@app.route('/facial_verification', methods=['GET', 'POST'])
def facial_verification():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'GET':
        return render_template('facial_verification.html')

    elif request.method == 'POST':
        try:
            username = session.get('username')
            user = User.query.filter_by(username=username).first()

            if not user:
                return jsonify({'success': False, 'message': 'Usuario no encontrado.'})

            if not user.profile_image:
                return jsonify({'success': False, 'message': 'Usuario no tiene malla facial registrada.'})

            # Obtener y validar la malla facial
            captured_mesh = request.json.get('captured_mesh')
            if not captured_mesh:
                return jsonify({'success': False, 'message': 'No se recibió la malla facial capturada.'})
            
            try:
                login_mesh_points = json.loads(captured_mesh)
            except json.JSONDecodeError:
                return jsonify({
                    'success': False,
                    'message': 'Error: Los datos recibidos no están en el formato JSON esperado.'
                })
            
            # Leer la malla facial almacenada
            try:
                stored_mesh_file_path = user.profile_image.replace('-', os.sep)
                full_file_path = os.path.join(app.root_path, stored_mesh_file_path.lstrip('/'))

                with open(full_file_path, 'r', encoding='utf-8') as mesh_file:
                    stored_mesh_points = json.load(mesh_file)
            except (FileNotFoundError, json.JSONDecodeError):
                app.logger.error(f"Error al leer malla facial almacenada para usuario {username}")
                return jsonify({
                    'success': False,
                    'message': 'Error al procesar los datos almacenados del usuario.'
                })

            # Validar y comparar mallas faciales
            if not validate_mesh_structure(login_mesh_points, stored_mesh_points):
                return jsonify({
                    'success': False,
                    'message': 'Estructura de malla facial inválida.'
                })

            similarity_score = compare_facial_meshes(login_mesh_points, stored_mesh_points)

            # Registrar el intento
            log_verification_attempt(
                user_id=user.id,
                success=similarity_score >= SIMILARITY_THRESHOLD,
                similarity_score=similarity_score
            )
    
            # Verificar similitud y proceder con el login
            if similarity_score >= SIMILARITY_THRESHOLD:
                login_user(user)
                session.pop('username', None)
                login_record = LoginRecord(
                    user_id=user.id,
                    login_datetime=datetime.now(),
                    ip_address=request.remote_addr,
                    user_agent=request.user_agent.string
                )
                db.session.add(login_record)
                db.session.commit()
                
                session.pop('username', None)
                return jsonify({
                    'success': True,
                    'message': 'Verificación facial exitosa.',
                    'redirect': url_for('user_inicio')
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Verificación facial fallida. Por favor, intente de nuevo.'
                })

        except Exception as e:
            app.logger.error(f"Error en verificación facial: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Error en la verificación. Por favor, intente de nuevo.'
            })

def filter_recent_logins():
    """
    Elimina registros duplicados de login dentro de un rango de 3 segundos,
    manteniendo solo el primer registro de cada grupo.
    """
    try:
        # Obtener todos los registros ordenados por fecha
        logins = LoginRecord.query.order_by(LoginRecord.login_datetime.desc()).all()
        
        to_delete = []
        last_login = None
        
        for login in logins:
            if last_login:
                # Si hay un login previo, verificar si está dentro del rango de 3 segundos
                time_diff = (last_login.login_datetime - login.login_datetime).total_seconds()
                if time_diff <= 3 and last_login.user_id == login.user_id:
                    to_delete.append(login)
            last_login = login
        
        # Eliminar los registros duplicados
        if to_delete:
            for login in to_delete:
                db.session.delete(login)
            db.session.commit()
            print(f"Se eliminaron {len(to_delete)} registros duplicados")
            
    except Exception as e:
        print(f"Error al filtrar logins: {str(e)}")
        db.session.rollback()

class LoginRecord(db.Model):
    __tablename__ = 'login_record'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    login_datetime = db.Column(db.DateTime, nullable=False)
    ip_address = db.Column(db.String(45))  # Falta este campo
    user_agent = db.Column(db.String(255))

    # Relación con el modelo User
    user = db.relationship('User', backref=db.backref('login_records', lazy=True))

def record_user_login(user):
    """
    Registra el inicio de sesión de un usuario
    
    Args:
        user: Instancia del modelo User que ha iniciado sesión
    """
    try:
        # Crear zona horaria para Perú
        peru_tz = pytz.timezone('America/Lima')
        
        # Obtener la hora actual en la zona horaria de Perú
        current_time = datetime.now(peru_tz)
        
        # Crear nuevo registro de inicio de sesión
        login_record = LoginRecord(
            user_id=user.id,
            login_datetime=current_time,
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string
        )
        
        # Guardar en la base de datos
        db.session.add(login_record)
        db.session.commit()
        
        # Registrar en el log del sistema
        app.logger.info(f"Login exitoso registrado para el usuario {user.username} at {current_time}")
        
    except Exception as e:
        # En caso de error, hacer rollback y registrar el error
        db.session.rollback()
        app.logger.error(f"Error al registrar el login del usuario {user.username}: {str(e)}")
        raise
    


@app.route('/login_history_detailed')
@login_required
@admin_required
def view_login_history_detailed():
    username = request.args.get('username')
    period = int(request.args.get('period', '7'))
    
    # Filtrar registros duplicados antes de mostrar
    filter_recent_logins()
    
    # Calcular fecha inicial
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period)
    
    # Preparar la consulta base
    query = LoginRecord.query.filter(LoginRecord.login_datetime.between(start_date, end_date))
    

    # Filtrar por usuario si se especificó
    if username:
        user = User.query.filter_by(username=username).first()
        if user:
            query = query.filter_by(user_id=user.id)
    
    # Obtener registros
    login_records = query.order_by(LoginRecord.login_datetime.desc()).all()
    
    # Calcular estadísticas
    stats = calculate_login_statistics(login_records)
    
    # Preparar datos para los gráficos
    daily_data = stats['daily_distribution']
    hourly_data = stats['hourly_distribution']
    
    # Obtener lista de usuarios para el selector
    all_users = User.query.order_by(User.username).all()
    
    return render_template(
        'login_history.html',  # Asegúrate de que este template existe
        login_records=login_records,
        stats=stats,
        all_users=all_users,
        selected_username=username,
        daily_labels=[d['date'] for d in daily_data],
        daily_data=[d['count'] for d in daily_data],
        hourly_labels=[f"{h:02d}:00" for h in range(24)],
        hourly_data=hourly_data
    )



def calculate_login_statistics(records):
    """Calcula estadísticas detalladas de los inicios de sesión."""
    stats = {
        'total_logins': len(records),
        'daily_average': 0,
        'peak_hour': '00:00',
        'daily_distribution': [],
        'hourly_distribution': [0] * 24
    }
    
    if not records:
        return stats
    
    # Agrupar por día
    daily_counts = {}
    hour_counts = [0] * 24
    
    for record in records:
        # Conteo diario
        date_str = record.login_datetime.strftime('%Y-%m-%d')
        daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
        
        # Conteo por hora
        hour = record.login_datetime.hour
        hour_counts[hour] += 1
    
    # Calcular promedios y picos
    stats['daily_average'] = len(records) / len(daily_counts) if daily_counts else 0
    peak_hour = hour_counts.index(max(hour_counts))
    stats['peak_hour'] = f"{peak_hour:02d}:00"
    
    # Preparar datos para gráficos
    stats['daily_distribution'] = [
        {'date': date, 'count': count}
        for date, count in sorted(daily_counts.items())
    ]
    
    stats['hourly_distribution'] = hour_counts
    
    return stats


def generate_login_statistics(login_records):
    """
    Genera estadísticas detalladas de los inicios de sesión.
    """
    stats = {
        'daily_average': 0,
        'most_active_day': 'N/A',
        'most_active_hour': 'N/A',
        'chart_data': {
            'hourly_distribution': [],
            'weekly_distribution': [],
            'monthly_trend': []
        },
        'insufficient_data': False
    }

    if not login_records:
        stats['insufficient_data'] = True
        return stats

    # Preparar contadores
    hourly_counts = defaultdict(int)
    weekday_counts = defaultdict(int)
    monthly_counts = defaultdict(int)
    daily_counts = defaultdict(int)

    # Procesar cada registro
    for record in login_records:
        date = record.login_datetime.date()
        hour = record.login_datetime.hour
        weekday = record.login_datetime.weekday()
        month = record.login_datetime.strftime('%Y-%m')

        daily_counts[date] += 1
        hourly_counts[hour] += 1
        weekday_counts[weekday] += 1
        monthly_counts[month] += 1

    # Verificar si hay suficientes días de datos
    if len(daily_counts) < 3:
        stats['insufficient_data'] = True
        
        # Calcular lo que se pueda con los datos disponibles
        if daily_counts:
            total_logins = sum(daily_counts.values())
            stats['daily_average'] = round(total_logins / len(daily_counts), 2)
            
            most_active_date = max(daily_counts.items(), key=lambda x: x[1])[0]
            stats['most_active_day'] = most_active_date.strftime('%Y-%m-%d')
            
            most_active_hour = max(hourly_counts.items(), key=lambda x: x[1])[0]
            stats['most_active_hour'] = f"{most_active_hour:02d}:00"

    # Si hay 3 o más días de datos
    if not stats['insufficient_data']:
        total_days = (max(daily_counts.keys()) - min(daily_counts.keys())).days + 1
        total_logins = sum(daily_counts.values())
        stats['daily_average'] = round(total_logins / max(total_days, 1), 2)
        
        most_active_date = max(daily_counts.items(), key=lambda x: x[1])[0]
        stats['most_active_day'] = most_active_date.strftime('%Y-%m-%d')
        
        most_active_hour = max(hourly_counts.items(), key=lambda x: x[1])[0]
        stats['most_active_hour'] = f"{most_active_hour:02d}:00"

    # Preparar datos para los gráficos
    # 1. Distribución por hora
    stats['chart_data']['hourly_distribution'] = {
        'labels': [f"{hour:02d}:00" for hour in range(24)],
        'data': [hourly_counts[hour] for hour in range(24)],
        'backgroundColor': 'rgba(59, 130, 246, 0.5)',
        'borderColor': 'rgb(59, 130, 246)'
    }

    # 2. Distribución semanal
    days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    stats['chart_data']['weekly_distribution'] = {
        'labels': days,
        'data': [weekday_counts[day] for day in range(7)],
        'backgroundColor': 'rgba(16, 185, 129, 0.5)',
        'borderColor': 'rgb(16, 185, 129)'
    }

    # 3. Tendencia mensual
    current_date = datetime.now()
    months = []
    month_data = []
    for i in range(5, -1, -1):
        date = current_date - timedelta(days=30*i)
        month = date.strftime('%Y-%m')
        months.append(date.strftime('%B %Y'))
        month_data.append(monthly_counts[month])

    stats['chart_data']['monthly_trend'] = {
        'labels': months,
        'data': month_data,
        'borderColor': 'rgb(99, 102, 241)',
        'fill': False
    }

    return stats

def generate_global_login_statistics(login_records):
    """
    Genera estadísticas detalladas de los inicios de sesión para todos los usuarios.
    """
    stats = {
        'global_stats': {
            'total_logins': 0,
            'daily_average': 0,
            'most_active_day': 'N/A',
            'most_active_hour': 'N/A',
            'most_active_user': 'N/A',
            'insufficient_data': False,
            'chart_data': {
                'hourly_distribution': [],
                'weekly_distribution': [],
                'monthly_trend': [],
                'user_distribution': []
            }
        },
        'user_stats': {}  # Estadísticas individuales por usuario
    }

    if not login_records:
        stats['global_stats']['insufficient_data'] = True
        return stats

    # Preparar contadores globales
    hourly_counts = defaultdict(int)
    weekday_counts = defaultdict(int)
    monthly_counts = defaultdict(int)
    daily_counts = defaultdict(int)
    user_counts = defaultdict(int)
    
    # Contadores por usuario
    user_daily_counts = defaultdict(lambda: defaultdict(int))
    user_hourly_counts = defaultdict(lambda: defaultdict(int))
    
    # Procesar cada registro
    for record in login_records:
        date = record.login_datetime.date()
        hour = record.login_datetime.hour
        weekday = record.login_datetime.weekday()
        month = record.login_datetime.strftime('%Y-%m')
        username = record.user.username

        # Conteos globales
        daily_counts[date] += 1
        hourly_counts[hour] += 1
        weekday_counts[weekday] += 1
        monthly_counts[month] += 1
        user_counts[username] += 1
        
        # Conteos por usuario
        user_daily_counts[username][date] += 1
        user_hourly_counts[username][hour] += 1

    # Verificar si hay suficientes días de datos
    if len(daily_counts) < 3:
        stats['global_stats']['insufficient_data'] = True
        
        # Calcular lo que se pueda con los datos disponibles
        if daily_counts:
            total_logins = sum(daily_counts.values())
            stats['global_stats']['total_logins'] = total_logins
            stats['global_stats']['daily_average'] = round(
                total_logins / len(daily_counts), 
                2
            )
            
            most_active_date = max(daily_counts.items(), key=lambda x: x[1])[0]
            stats['global_stats']['most_active_day'] = most_active_date.strftime('%Y-%m-%d')
            
            most_active_hour = max(hourly_counts.items(), key=lambda x: x[1])[0]
            stats['global_stats']['most_active_hour'] = f"{most_active_hour:02d}:00"
            
            most_active_user = max(user_counts.items(), key=lambda x: x[1])[0]
            stats['global_stats']['most_active_user'] = most_active_user

    # Si hay 3 o más días de datos
    if not stats['global_stats']['insufficient_data']:
        total_days = (max(daily_counts.keys()) - min(daily_counts.keys())).days + 1
        stats['global_stats']['total_logins'] = sum(daily_counts.values())
        stats['global_stats']['daily_average'] = round(
            stats['global_stats']['total_logins'] / max(total_days, 1), 
            2
        )
        
        # Día más activo
        most_active_date = max(daily_counts.items(), key=lambda x: x[1])[0]
        stats['global_stats']['most_active_day'] = most_active_date.strftime('%Y-%m-%d')
        
        # Hora más activa
        most_active_hour = max(hourly_counts.items(), key=lambda x: x[1])[0]
        stats['global_stats']['most_active_hour'] = f"{most_active_hour:02d}:00"
        
        # Usuario más activo
        most_active_user = max(user_counts.items(), key=lambda x: x[1])[0]
        stats['global_stats']['most_active_user'] = most_active_user

    # Preparar datos para los gráficos globales
    # 1. Distribución por hora
    stats['global_stats']['chart_data']['hourly_distribution'] = {
        'labels': [f"{hour:02d}:00" for hour in range(24)],
        'data': [hourly_counts[hour] for hour in range(24)],
        'backgroundColor': 'rgba(59, 130, 246, 0.5)',
        'borderColor': 'rgb(59, 130, 246)'
    }

    # 2. Distribución semanal
    days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    stats['global_stats']['chart_data']['weekly_distribution'] = {
        'labels': days,
        'data': [weekday_counts[day] for day in range(7)],
        'backgroundColor': 'rgba(16, 185, 129, 0.5)',
        'borderColor': 'rgb(16, 185, 129)'
    }

    # 3. Tendencia mensual
    current_date = datetime.now()
    months = []
    month_data = []
    for i in range(5, -1, -1):
        date = current_date - timedelta(days=30*i)
        month = date.strftime('%Y-%m')
        months.append(date.strftime('%B %Y'))
        month_data.append(monthly_counts[month])

    stats['global_stats']['chart_data']['monthly_trend'] = {
        'labels': months,
        'data': month_data,
        'borderColor': 'rgb(99, 102, 241)',
        'fill': False
    }

    # 4. Distribución por usuario (top 10 usuarios más activos)
    top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    stats['global_stats']['chart_data']['user_distribution'] = {
        'labels': [user[0] for user in top_users],
        'data': [user[1] for user in top_users],
        'backgroundColor': 'rgba(124, 58, 237, 0.5)',
        'borderColor': 'rgb(124, 58, 237)'
    }

    # Generar estadísticas por usuario
    for username in user_counts.keys():
        user_dates = user_daily_counts[username].keys()
        if user_dates:
            total_user_days = len(user_dates)
            total_user_logins = user_counts[username]
            
            # Hora más activa del usuario
            user_peak_hour = max(user_hourly_counts[username].items(), key=lambda x: x[1])[0]
            
            stats['user_stats'][username] = {
                'total_logins': total_user_logins,
                'daily_average': round(total_user_logins / max(total_user_days, 1), 2),
                'peak_hour': f"{user_peak_hour:02d}:00",
                'percentage_of_total': round(
                    (total_user_logins / stats['global_stats']['total_logins']) * 100, 
                    2
                )
            }

    return stats

# Ruta para ver las estadísticas globales de inicio de sesión
@app.route('/admin/global_login_statistics', methods=['GET'])
@login_required
@admin_required
def view_global_login_statistics():
    """
    Vista administrativa para mostrar estadísticas globales de inicio de sesión
    de todos los usuarios. Solo accesible por administradores.
    """
    try:
        # Obtener todos los registros de login
        login_records = LoginRecord.query.all()
        
        # Generar estadísticas
        stats = generate_global_login_statistics(login_records)
        
        # Si es una solicitud AJAX, devolver JSON
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': True,
                'data': stats
            })
        
        # Si no es AJAX, renderizar template
        return render_template(
            'global_login_statistics.html',
            stats=stats,
            insufficient_data=stats['global_stats']['insufficient_data']
        )
        
    except Exception as e:
        app.logger.error(f"Error al generar estadísticas globales de login: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error al generar las estadísticas globales de inicio de sesión.'
        }), 500

@app.route('/user_inicio')
@login_required  # Asegura que solo usuarios autenticados puedan acceder
def user_inicio():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    # Datos para el carrusel
    slides = [
        {
            "title": "KIRAFACE",
            "description": "Autenticación biométrica rápida y segura que usa reconocimiento facial y detección de parpadeo para garantizar acceso sin contacto con precisión y rapidez en tiempo real.",
            "image": "images/car1.jpeg"
        },
        {
            "title": "Seguridad Avanzada",
            "description": "Sistema de seguridad biométrico de última generación",
            "image": "images/car2.jpeg"
        },
        {
            "title": "Acceso Sin Contacto",
            "description": "Tecnología moderna para control de acceso",
            "image": "images/car3.jpeg"
        },
        {
            "title": "Biometria",
            "description": "Seguridad biometrica que garantiza presicion y rapidez",
            "image": "images/car4.jpeg"
        }
    ]
    
    # Renderiza la plantilla con los datos del carrusel
    return render_template('user_inicio.html', slides=slides)

@app.route('/recover_account', methods=['GET', 'POST'])
def recover_account():
    if request.method == 'POST':
        username = request.form.get('username')
        phone_number = request.form.get('phone_number')
        respuesta = request.form.get('respuesta')
        
        user = User.query.filter_by(username=username).first()
        
        if not user:
            flash('Usuario no encontrado')
            return redirect(url_for('recover_account'))
        
        if user.phone_number != phone_number:
            flash('Número de teléfono incorrecto')
            return redirect(url_for('recover_account'))
        
        if user.respuesta.lower() != respuesta.lower():
            flash('Respuesta de seguridad incorrecta')
            return redirect(url_for('recover_account'))
        
        # Si todas las validaciones son correctas, generar token
        token = generate_reset_token(username)
        session['reset_token'] = token
        session['reset_username'] = username
        return redirect(url_for('reset_password', token=token))
            
    return render_template('recover_account.html')

@app.route('/get_security_question/<username>')
def get_security_question(username):
    user = User.query.filter_by(username=username).first()
    if user:
        questions = {
            1: "¿Cuál es el nombre de tu primera mascota?",
            2: "¿En qué ciudad naciste?",
            3: "¿Cuál es el nombre de tu madre?",
            4: "¿Cuál fue tu primer colegio?",
            5: "¿Cuál es tu comida favorita?"
        }
        return jsonify({
            'question': questions.get(user.num_preg, "Pregunta no encontrada")
        })
    return jsonify({'question': None})

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if 'reset_token' not in session or session['reset_token'] != token:
        flash('Sesión inválida o expirada')
        return redirect(url_for('recover_account'))
        
    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if new_password != confirm_password:
            flash('Las contraseñas no coinciden')
            return redirect(url_for('reset_password', token=token))
            
        username = session.get('reset_username')
        user = User.query.filter_by(username=username).first()
        
        if user:
            user.password = generate_password_hash(new_password)
            db.session.commit()
            session.pop('reset_token', None)
            session.pop('reset_username', None)
            flash('Contraseña actualizada correctamente')
            return redirect(url_for('login'))
        
    return render_template('reset_password.html')

def generate_reset_token(username):
    # Genera un token único para resetear la contraseña
    return secrets.token_urlsafe(32)

def verify_reset_token(token):
    # Verifica el token y retorna el username asociado
    if 'reset_token' in session and session['reset_token'] == token:
        return session.get('reset_username')
    return None

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Has cerrado sesión correctamente.')
    return redirect(url_for('login')) 

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)