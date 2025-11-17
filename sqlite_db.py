import sqlite3
import json
import os
import sys
import base64
import numpy as np
import cv2
import traceback

# Caminho do banco de dados
DB_PATH = os.path.join(os.path.dirname(__file__), 'empathia.db')

def init_database():
    """Inicializar banco de dados SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Tabela de funcionários (COM FOTO EM BASE64)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                role TEXT,
                department TEXT,
                email TEXT,
                embedding TEXT,
                facial_area TEXT,
                face_confidence REAL,
                photo_base64 TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de logs de emoção
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotion_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER NOT NULL,
                employee_name TEXT,
                dominant_emotion TEXT,
                emotions TEXT,
                confidence REAL,
                analysis_duration INTEGER,
                samples_collected INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"✅ Banco SQLite inicializado: {DB_PATH}")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao inicializar banco: {e}")
        return False

def check_mongodb():
    """Verificar se banco está disponível"""
    if not os.path.exists(DB_PATH):
        return init_database()
    return True

def save_employee_data(employee_data):
    """Salvar funcionário COM FOTO EM BASE64"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Converter dados para JSON
        embedding_json = json.dumps(employee_data.get('embedding', []))
        facial_area_json = json.dumps(employee_data.get('facial_area', {}))
        
        # ✅ Converter foto para Base64
        photo_base64 = None
        if employee_data.get('photo') is not None:
            photo_base64 = image_to_base64(employee_data['photo'])
        
        cursor.execute('''
            INSERT INTO employees (name, role, department, email, embedding, facial_area, face_confidence, photo_base64)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            employee_data.get('name'),
            employee_data.get('role'),
            employee_data.get('department'),
            employee_data.get('email'),
            embedding_json,
            facial_area_json,
            employee_data.get('face_confidence', 0.0),
            photo_base64  # ✅ SALVAR COMO BASE64
        ))
        
        employee_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Funcionário cadastrado com ID: {employee_id}")
        print(f"   📍 Pontos faciais salvos: {employee_data.get('facial_area')}")
        print(f"   📷 Foto Base64: {'Sim' if photo_base64 else 'Não'} ({len(photo_base64) if photo_base64 else 0} chars)")
        return str(employee_id)
        
    except Exception as e:
        print(f"❌ Erro ao salvar: {e}")
        traceback.print_exc()
        return None

def get_all_employees():
    """Buscar todos os funcionários COM FOTO BASE64"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM employees')
        rows = cursor.fetchall()
        
        employees = []
        for row in rows:
            # ✅ DEBUG: Mostrar estrutura da row
            print(f"\n🔍 DEBUG - Estrutura da row:")
            print(f"   Total de campos: {len(row)}")
            for i, field in enumerate(row):
                field_preview = str(field)[:50] if field else "NULL"
                print(f"   [{i}] = {field_preview}")
            
            # ✅ Converter Base64 para imagem
            photo = None
            photo_base64 = row[8] if len(row) > 8 else None
            
            print(f"\n   photo_base64 está no índice [8]: {photo_base64 is not None}")
            
            if photo_base64:
                print(f"   Tamanho do Base64: {len(photo_base64)} chars")
                photo = base64_to_image(photo_base64)
                
                if photo is not None:
                    print(f"   ✅ Imagem decodificada: {photo.shape}")
                else:
                    print(f"   ❌ Falha ao decodificar Base64")
            else:
                print(f"   ⚠️ photo_base64 está vazio ou NULL")
            
            employee = {
                '_id': str(row[0]),
                'name': row[1],
                'role': row[2],
                'department': row[3],
                'email': row[4],
                'embedding': json.loads(row[5]) if row[5] else [],
                'facial_area': json.loads(row[6]) if row[6] else {},
                'face_confidence': row[7],
                'photo': photo,  # ✅ IMAGEM OPENCV
                'photo_base64': photo_base64,  # ✅ TAMBÉM RETORNAR BASE64
                'created_at': row[9] if len(row) > 9 else None
            }
            
            print(f"\n   📦 Dicionário final:")
            print(f"      photo: {employee['photo'] is not None}")
            print(f"      photo_base64: {employee['photo_base64'] is not None}")
            
            employees.append(employee)
        
        conn.close()
        return employees
        
    except Exception as e:
        print(f"❌ Erro ao buscar: {e}")
        traceback.print_exc()
        return []
    
def find_employee_by_face(face_embedding, facial_area=None):
    """
    ✅ BUSCA OTIMIZADA - Reconhece com óculos/barba/variações
    Threshold reduzido para 70% (mais flexível)
    """
    try:
        from utils.face_recognition import FaceRecognitionSystem
        
        employees = get_all_employees()
        
        if not employees:
            print("⚠️ Nenhum funcionário cadastrado")
            return None
        
        # ✅ VALIDAR TAMANHO DO EMBEDDING
        current_embedding_size = len(face_embedding)
        if current_embedding_size != 512:
            print(f"❌ Embedding atual tem {current_embedding_size} dimensões (esperado: 512)")
            return None
        
        print(f"\n🔍 INICIANDO RECONHECIMENTO FACIAL OTIMIZADO")
        print(f"📊 Comparando com {len(employees)} funcionários...")
        print(f"⚙️ Threshold: 70% (aceita variações)")
        
        best_score = 0.0
        best_match = None
        all_scores = []
        
        for employee in employees:
            if not employee.get('embedding'):
                print(f"⚠️ {employee.get('name')}: sem embedding")
                continue
            
            try:
                stored_embedding = employee['embedding']
                stored_area = employee.get('facial_area', {})
                
                # ✅ VALIDAR COMPATIBILIDADE
                if len(stored_embedding) != len(face_embedding):
                    print(f"⚠️ {employee.get('name')}: embedding incompatível")
                    print(f"   ⚠️ RECADASTRE ESSE FUNCIONÁRIO!")
                    continue
                
                # ✅ COMPARAÇÃO OTIMIZADA (retorna apenas score)
                score_final = FaceRecognitionSystem.comparar_faces_hibrido_otimizado(
                    face_embedding,
                    facial_area if facial_area else {},
                    stored_embedding,
                    stored_area
                )
                
                all_scores.append({
                    'name': employee.get('name'),
                    'score': score_final
                })
                
                print(f"   👤 {employee.get('name')}: {score_final:.2%}")
                
                if score_final > best_score:
                    best_score = score_final
                    best_match = employee
                    
            except Exception as e:
                print(f"⚠️ Erro ao comparar com {employee.get('name')}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # ✅ THRESHOLD AJUSTADO: 75% (mais rigoroso)
        THRESHOLD = 0.75 
        
        if best_match and best_score >= THRESHOLD:
            is_match, quality = FaceRecognitionSystem.verificar_match(best_score, THRESHOLD)
            
            print(f"\n✅ MATCH ENCONTRADO!")
            print(f"   Nome: {best_match.get('name')}")
            print(f"   Score: {best_score:.2%}")
            print(f"   Qualidade: {quality}")
            print(f"   Threshold: {THRESHOLD:.0%}")
            
            # Top 3 para debug
            print(f"\n📊 Top 3 scores:")
            top_3 = sorted(all_scores, key=lambda x: x['score'], reverse=True)[:3]
            for i, item in enumerate(top_3, 1):
                print(f"   {i}. {item['name']}: {item['score']:.2%}")
            
            return best_match
        else:
            print(f"\n❌ NÃO RECONHECIDO")
            if best_match:
                print(f"   📊 Melhor match: {best_match.get('name')}")
                print(f"   📊 Score: {best_score:.2%}")
                print(f"   ⚠️ Abaixo do threshold: {THRESHOLD:.0%}")
            
            # Top 3 para debug
            print(f"\n📊 Top 3 scores:")
            top_3 = sorted(all_scores, key=lambda x: x['score'], reverse=True)[:3]
            for i, item in enumerate(top_3, 1):
                print(f"   {i}. {item['name']}: {item['score']:.2%}")
            
            return None
        
    except Exception as e:
        print(f"❌ Erro ao buscar por face: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def update_employee_photo(employee_id, photo, embedding, facial_area, confidence):
    """Atualizar foto e dados biométricos (COM BASE64)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Converter para JSON
        embedding_json = json.dumps(embedding)
        facial_area_json = json.dumps(facial_area)
        
        # ✅ Converter foto para Base64
        photo_base64 = image_to_base64(photo)
        
        cursor.execute('''
            UPDATE employees 
            SET photo_base64 = ?,
                embedding = ?, 
                facial_area = ?, 
                face_confidence = ?
            WHERE id = ?
        ''', (
            photo_base64,
            embedding_json,
            facial_area_json,
            confidence,
            employee_id
        ))
        
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if updated:
            print(f"✅ Foto atualizada para funcionário {employee_id}")
        else:
            print(f"⚠️ Funcionário {employee_id} não encontrado")
        
        return updated
        
    except Exception as e:
        print(f"❌ Erro ao atualizar foto: {e}")
        traceback.print_exc()
        return False

def image_to_base64(image):
    """Converter imagem OpenCV para Base64"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f"❌ Erro ao converter para Base64: {e}")
        return None

def base64_to_image(base64_string):
    """Converter Base64 para imagem OpenCV"""
    try:
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"❌ Erro ao converter Base64: {e}")
        return None

def save_emotion_log(employee_id, emotion_data):
    """Salvar log de emoção"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        emotions_json = json.dumps(emotion_data.get('emotions', {}))
        
        cursor.execute('''
            INSERT INTO emotion_logs 
            (employee_id, employee_name, dominant_emotion, emotions, 
             confidence, analysis_duration, samples_collected)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            employee_id,
            emotion_data.get('name'),
            emotion_data.get('dominant_emotion'),
            emotions_json,
            emotion_data.get('confidence'),
            emotion_data.get('duration', 5),
            emotion_data.get('samples_collected', 0)
        ))
        
        log_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Log salvo: {log_id}")
        return str(log_id)
        
    except Exception as e:
        print(f"❌ Erro ao salvar log: {e}")
        return None

def get_employee_emotion_history(employee_id):
    """Buscar histórico de emoções"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM emotion_logs 
            WHERE employee_id = ? 
            ORDER BY timestamp DESC
        ''', (employee_id,))
        
        rows = cursor.fetchall()
        
        logs = []
        for row in rows:
            log = {
                '_id': str(row[0]),
                'employee_id': str(row[1]),
                'employee_name': row[2],
                'dominant_emotion': row[3],
                'emotions': json.loads(row[4]) if row[4] else {},
                'confidence': row[5],
                'analysis_duration': row[6],
                'samples_collected': row[7],
                'timestamp': row[8]
            }
            logs.append(log)
        
        conn.close()
        return logs
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return []

def get_recent_emotion_logs(limit=10):
    """Buscar logs recentes"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM emotion_logs 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        
        logs = []
        for row in rows:
            log = {
                '_id': str(row[0]),
                'employee_id': str(row[1]),
                'employee_name': row[2],
                'dominant_emotion': row[3],
                'emotions': json.loads(row[4]) if row[4] else {},
                'confidence': row[5],
                'analysis_duration': row[6],
                'samples_collected': row[7],
                'timestamp': row[8]
            }
            logs.append(log)
        
        conn.close()
        return logs
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return []

def delete_employee(employee_id):
    """Deletar funcionário"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Deletar logs
        cursor.execute('DELETE FROM emotion_logs WHERE employee_id = ?', (employee_id,))
        
        # Deletar funcionário
        cursor.execute('DELETE FROM employees WHERE id = ?', (employee_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if deleted:
            print(f"✅ Funcionário {employee_id} deletado")
        else:
            print(f"⚠️ Funcionário {employee_id} não encontrado")
        
        return deleted
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

# Inicializar banco ao importar
print("="*60)
print("INICIALIZANDO BANCO DE DADOS LOCAL (SQLite)")
print("="*60)
init_database()
print("="*60)