import socket
import time
import pychromecast
from gtts import gTTS


def get_speaker(ip_addr=None, name=None):
    if ip_addr:
        return pychromecast.Chromecast(str(ip_addr))

    speakers = pychromecast.get_chromecasts()
    if len(speakers) == 0:
        print("No devices are found")
        raise Exception
    if name:
        return next(s for s in speakers if s.device.friendly_name == name)
    return next(speakers)


def speak(text, speaker, lang="ja"):
    try:
        #tts = gTTS(text=text, lang=lang)
        #urls = tts.get_urls()
        #print(urls[0])
        if not speaker.is_idle:
            print("Killing current running app")
            speaker.quit_app()
            time.sleep(5)
        speaker.wait()
        #print(urls[0])
        urls = "https://translate.google.com/translate_tts?idx=0&total=1&tl=ja&tk=845198.665787&textlen=3&client=tw-ob&q=%E5%AF%86%E3%81%A7%E3%81%99&ie=UTF-8&ttsspeed=1"
        # speaker.media_controller.play_media('sample.mp3', "audio/mp3")
        speaker.media_controller.play_media(urls, 'audit/mp3')
        speaker.media_controller.block_until_active()
    except Exception as error:
        print(str(error))
        raise Exception


def check_speaker(speaker, lang):
    try:
        #speak(text="OK", speaker=speaker, lang=lang)
        print("You are ready to speak!")
        return True
    except Exception as error:
        print("Try an another ip or name: %s" % (str(error)))
        return False


def prepare_speaker():
    print("Enter language (English: en or Japanese: ja): ", end="")
    #lang = input()
    lang="ja"
    #print("Enter Google Home name or IP: ", end="")
    #name_or_ip = input()
    name_or_ip = "172.20.11.176"
    try:
        socket.inet_aton(name_or_ip)
        speaker = get_speaker(ip_addr=name_or_ip)
    except socket.error:
        speaker = get_speaker(name=name_or_ip)
    except Exception as error:
        print("Error: %s" % (str(error)))
        raise Exception
    return speaker, lang


def googlehome():
    while True:
        try:
            speaker, lang = prepare_speaker()
        except Exception:
            continue
        if check_speaker(speaker, lang):
            break
        print("Failed to setup. Try again!")

    print("Start typing ...")
    text = ""
    while text != "bye":
        print(">> ", end="")
        #text = input()
        text="密です"
        if text:
            speak(text, speaker, lang)
            text="bye"


# if __name__ == "__main__":
#     main()
