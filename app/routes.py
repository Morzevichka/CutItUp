from app import app, db
from app.forms import LoginForm, RegisterForm
from flask import flash, redirect, request, url_for, render_template, send_from_directory, abort, jsonify
from flask_login import current_user, login_user, logout_user
import os
from app.entity.app_user import AppUser
from app.entity.chat import Chat
from app.entity.message import Message
from flask import session
from datetime import datetime
import uuid
from app.model.predictions import VideoTrimmer

@app.route('/process_video', methods=['POST'])
def process_video():
    if request.is_json:
        data = request.get_json()

        chat_id = data.get('chat_id')
        filename = data.get('filename')
        num_clips = int(data.get('numClips', 1))
        clip_duration = int(data.get('clipDuration', 10))
    else:
        return jsonify({"error": "Invalid content type"}), 400

    print(chat_id, filename, num_clips, clip_duration)

    if current_user.is_authenticated:
        video_names = VideoTrimmer().predict_trimmed(chat_id=chat_id, filename=filename, clips=num_clips, rec_clip_duration=clip_duration)
        for video in video_names:
            message = Message(filename=video, chat_id=chat_id, sender_type='model')
            db.session.add(message)
        db.session.commit()
    else:
        video_names = VideoTrimmer().predict_trimmed(chat_id=None, filename=filename, clips=num_clips, rec_clip_duration=clip_duration)
        chats = session.get('chats', [])
        for chat in chats:
            if chat['chat_id'] == chat_id:
                for video_name in video_names:
                    message = {'filename': video_name, 'chat_id': chat_id, 'sender_type': 'model', 'timestamp': datetime.utcnow()}
                    chat['messages'].append(message)
                break
        session['chats'] = chats
    return jsonify({"success": True, "chat_id": chat_id})

@app.route('/videos/<path:filepath>')
def load_video(filepath):
    video_dir = app.config['BASE_VIDEO_PATH']
    file_path = os.path.join(video_dir, filepath)

    if not os.path.isfile(file_path) or not file_path.endswith(('.mp4', '.avi', '.mov')):
        abort(404)

    return send_from_directory(video_dir, filepath)

@app.route("/")
@app.route("/index")
def index():
    if current_user.is_authenticated:
        chats = Chat.query.filter_by(user_id=current_user.id).all()
    else:
        chats = session.get('chats', [])
    return render_template('index.html', chats=chats)

@app.route('/upload_video/', methods=['POST'])
@app.route('/upload_video/<chat_id>', methods=['POST'])
def upload_video(chat_id=None):
    if 'video' not in request.files:
        flash('No file selected', 'danger')
        return redirect(url_for('index'))
    
    video = request.files['video']
    if not video:
        flash('No selected file', 'danger')
        return redirect(url_for('index'))
    
    filename = video.filename

    if current_user.is_authenticated:
        if chat_id is None:
            chat_id = str(uuid.uuid4())

        chat = Chat.query.get(chat_id)
        if not chat:
            chat = Chat(id=chat_id, name=filename, user_id=current_user.id)
            db.session.add(chat)
            db.session.commit()
        
        chat_video_folder = os.path.join(app.config['BASE_VIDEO_PATH'], 'chats', chat_id)
        os.makedirs(chat_video_folder, exist_ok=True)
        file_path = os.path.join(chat_video_folder, filename)

        video.save(file_path)

        message = Message(filename=filename, chat_id=chat_id, sender_type='user')
        db.session.add(message)
        db.session.commit()

        flash('Video uploaded successfully', 'success') 

    else:
        if chat_id is None:
            chat_id = str(uuid.uuid4())

        videos_folder = app.config['BASE_VIDEO_PATH']
        file_path = os.path.join(videos_folder, filename)

        os.makedirs(videos_folder, exist_ok=True)
        video.save(file_path)

        chat_data = {'chat_id': chat_id, 'name': filename, 'user_id': None, 'messages': []}

        chats = session.get('chats', [])
        chat_found = False
        for chat in chats:
            if chat['chat_id'] == chat_id:
                chat_found = True
                chat['messages'].append({'filename': filename, 'timestamp': datetime.utcnow(), 'sender_type': 'user'})
                break

        if not chat_found:
            chat_data['messages'].append({'filename': filename, 'timestamp': datetime.utcnow(), 'sender_type': 'user'})
            chats.append(chat_data)

        session['chats'] = chats
        flash('Video uploaded successfully', 'success')

    return jsonify({
        'success': True,
        'chat_id': chat_id,
        'filename': filename
    })

@app.route('/c/<chat_id>', methods=['GET', 'POST'])
def chat(chat_id):
    if current_user.is_authenticated:
        chats = Chat.query.filter_by(user_id=current_user.id).all()
        messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.timestamp).all()
    else:
        chats = session.get('chats', [])
        messages = []
        for chat in chats:
            if chat['chat_id'] == chat_id:
                messages = chat['messages']
                break

        messages = sorted(messages, key=lambda m: m['timestamp'])

    return render_template('chat.html', chats=chats, chat_id=chat_id, messages=messages)

@app.route('/get_latest_messages/<chat_id>')
def get_latest_messages(chat_id):
    if current_user.is_authenticated:
        message = Message.query.filter_by(chat_id=chat_id).order_by(Message.timestamp.desc()).first()
        if message and message.sender_type == 'model':
            return jsonify({'messagesUpdated': True})
    else:
        chats = session.get('chats', [])
        messages = []
        
        for chat in chats:
            if chat['chat_id'] == chat_id:
                messages = chat['messages']
                break
        
        last_message = messages[-1] if messages else None
        if last_message and last_message['sender_type'] == 'model':
            return jsonify({'messagesUpdated': True})

    return jsonify({'messagesUpdated': False})

@app.route('/search')
def search():
    return render_template('index.html')

@app.route('/rename_chat')
def rename_chat():
    return render_template('index.html')

@app.route('/delete_chat/<chat_id>', methods=['GET'])
def delete_chat(chat_id):
    if current_user.is_authenticated:
        chat_to_delete = Chat.query.get(chat_id)
        if not chat_to_delete:
            flash(f'Chat hasn\'t found (maybe deleted)', 'danger')
            return redirect(url_for('index'))
        

        db.session.delete(chat_to_delete)
        db.session.commit()
        flash('Chat has been deleted', 'success')
    else:
        if 'chats' not in session:
            flash(f'Chat hasn\'t found (maybe deleted)', 'danger')
            return redirect(url_for('index'))
        
        session['chats'] = [chat for chat in session['chats'] if chat['chat_id'] != chat_id]
    return redirect(url_for('index'))


@app.route("/singup", methods=["POST", "GET"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        if AppUser.query.filter_by(username=form.username.data).first():
            flash('Username is already taken. Please choose a different one.', 'danger')
            return redirect(url_for('register'))
        
        user = AppUser(username=form.username.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash(f'Successful Sign up! Your Id is {user.id}', 'success')
        login_user(user)
        return redirect(url_for("index"))
    return render_template("register.html", title='Sing up', form=form)

@app.route("/login", methods=["POST", "GET"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    
    form = LoginForm()
    if form.validate_on_submit(): 
        user = AppUser.query.filter_by(username=form.username.data).first()
        
        if user is None or not user.check_password(form.password.data):
            flash('Incorrect login or password', 'warning')
            return redirect(url_for("login"))
        
        login_user(user, remember=form.remember_me.data)
        flash('Seccussful Log In', 'success')
        return redirect(url_for("index"))
    return render_template('login.html', title="Log in", form=form)

@app.route("/logout")
def logout():
    logout_user()
    flash('You have logged out', 'warning')
    return redirect(url_for('index'))

@app.route('/clear_session')
def clear_session():
    session.clear()
    return redirect(url_for('index'))