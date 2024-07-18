# views.py
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .forms import UploadForm, NumberOfPeopleForm
from pyannote.audio import Pipeline
import os
import soundfile as sf
import numpy as np
import openai
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.conf import settings
from elevenlabs.client import ElevenLabs, VoiceSettings
import logging
import base64
from django.views.decorators.http import require_http_methods

logging.basicConfig(level=logging.INFO)
openai.api_key = 'sk-proj-XrzlCD7GRdhbQpwEvbGUT3BlbkFJ4TSw8sDEANJvR5RgAhhv'
elevenlabs_api_key = '866f355f0c138130ff9afa77e40041bf'

def select_number_of_people(request):
    if request.method == 'POST':
        form = NumberOfPeopleForm(request.POST)
        if form.is_valid():
            number_of_people = form.cleaned_data['number_of_people']
            request.session['number_of_people'] = number_of_people
            return redirect('voice_separation')
    else:
        form = NumberOfPeopleForm()
    return render(request, 'select_number_of_people.html', {'form': form})

def voice_separation(request):
    context = {'form': UploadForm()}  # 초기 컨텍스트 설정

    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)

            number_of_people = request.session.get('number_of_people', '2')

            try:
                # pyannote.audio 파이프라인 로드
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token="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"
                )

                # 오디오 파일을 불러옵니다.
                audio_data, sample_rate = sf.read(file_path)
                # 화자 분할을 실행합니다.
                diarization = diarization_pipeline({'audio': file_path})

                speakers_audio = {}
                # 화자별로 음성 데이터를 분리합니다.
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start_sample = int(turn.start * sample_rate)
                    end_sample = int(turn.end * sample_rate)
                    speaker_audio = audio_data[start_sample:end_sample]
                    # 화자별로 음성 데이터를 저장합니다.
                    if speaker in speakers_audio:
                        speakers_audio[speaker].append(speaker_audio)
                    else:
                        speakers_audio[speaker] = [speaker_audio]

                # 화자별로 오디오 파일을 저장합니다.
                spleeter_output_path = os.path.join(fs.location, os.path.splitext(file.name)[0])
                os.makedirs(spleeter_output_path, exist_ok=True)
                for speaker, audio in speakers_audio.items():
                    # 모든 발화를 하나의 오디오 트랙으로 연결합니다.
                    speaker_audio = np.concatenate(audio, axis=0)
                    # 오디오 파일로 저장합니다.
                    speaker_output_path = os.path.join(spleeter_output_path, f'speaker_{speaker}.wav')
                    sf.write(speaker_output_path, speaker_audio, sample_rate)

                    # 파일 크기 확인
                    print(f"파일 {speaker_output_path} 크기: {os.path.getsize(speaker_output_path)} 바이트")

                # 화자 파일 경로를 세션에 저장
                request.session['speaker_files'] = [os.path.join(spleeter_output_path, f) for f in os.listdir(spleeter_output_path) if f.startswith('speaker')]

            except Exception as e:
                # 오류 메시지를 컨텍스트에 추가합니다.
                context['error_message'] = str(e)
                print(f"분리 실패: {str(e)}")

            # 처리 후 form 초기화
            context['form'] = UploadForm()
            return redirect('select_speaker')  # 음성 파일 선택 페이지로 리디렉션

    return render(request, 'upload.html', context)

def select_speaker(request):
    # 세션에서 화자 파일 경로를 불러옴
    speaker_files = request.session.get('speaker_files', [])

    speakers = []
    for idx, speaker_file in enumerate(speaker_files):
        file_url = f"{settings.MEDIA_URL}{speaker_file.split(settings.MEDIA_ROOT)[-1]}"
        speakers.append({'id': idx, 'file': file_url})

        if os.path.exists(speaker_file):
            print(f"파일 {speaker_file} 존재함")
            print(f"파일 {speaker_file} 크기: {os.path.getsize(speaker_file)} 바이트")
        else:
            print(f"파일 {speaker_file} 존재하지 않음")

    if request.method == 'POST':
        selected_speaker = request.POST.get('selected_speaker')
        request.session['selected_speaker'] = selected_speaker
        return redirect('query_view')

    return render(request, 'select_speaker.html', {'speakers': speakers})

# ElevenLabs 클라이언트 초기화
client = ElevenLabs(api_key=elevenlabs_api_key)

def get_completion(prompt, past_messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=past_messages + [{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    answer = response['choices'][0]['message']['content']
    return answer

def text_to_speech(text: str) -> bytes:
    try:
        response = client.text_to_speech.convert(
            voice_id="HIfeW6Vd8E8LZ3HAkZk8",
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5, similarity_boost=0.8, style=0.0, use_speaker_boost=True
            ),
        )
        audio_data = b"".join(chunk for chunk in response)
        return audio_data
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return None

@csrf_exempt
def query_view(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        prompt = str(prompt)
        if 'past_messages' not in request.session:
            request.session['past_messages'] = []
        text_response = get_completion(prompt, request.session['past_messages'])
        request.session['past_messages'].append({"role": "system", "content": text_response})
        request.session.modified = True
        audio_data = text_to_speech(text_response)
        audio_data_base64 = base64.b64encode(audio_data).decode('utf-8')
        return JsonResponse({'response': text_response, 'audio_data': audio_data_base64})
    return render(request, 'chat.html')

@require_http_methods(["POST"])
def summarize_text(request):
    data = request.POST.get('text', '')
    if not data:
        return JsonResponse({'error': 'No text provided'}, status=400)

    openai.api_key = os.getenv("YOUR_OPENAI_API_KEY")

    response = openai.Completion.create(
      engine="davinci", 
      prompt=f"요약: {data}\n\n###\n\n",
      temperature=0.5,
      max_tokens=150,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )

    summarized_text = response.choices[0].text.strip()

    return JsonResponse({'original': data, 'summary': summarized_text})

def home(request):
    return render(request, 'home.html')

def choose(request):
    return render(request, 'choose.html')

def make(request):
    return render(request, 'make.html')

def choose_ch1(request):
    return render(request, 'choose_ch1.html')

def chat(request):
    return render(request, 'chat.html')

def make_ch1(request):
    return render(request, 'make_ch1.html')
