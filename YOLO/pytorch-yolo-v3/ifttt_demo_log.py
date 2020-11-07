import requests

# IFTTT_Webhook
def ifttt_webhook(s_type):
    if s_type==1:
        payload = {"value1": "密を検出しました",
                    "value2": "密閉",
                    "value3": "" }
        url = "https://maker.ifttt.com/trigger/" + "Cs_log" + "/with/key/bPjDX0mOvggkBC15P93cy8"
        response = requests.post(url, data=payload)
    elif s_type==2:
        payload = {"value1": "密を検出しました",
                    "value2": "密集",
                    "value3": "" }
        url = "https://maker.ifttt.com/trigger/" + "Cs_log" + "/with/key/bPjDX0mOvggkBC15P93cy8"
        response = requests.post(url, data=payload)
    elif s_type==3:
        payload = {"value1": "密を検出しました",
                    "value2": "密接",
                    "value3": "" }
        url = "https://maker.ifttt.com/trigger/" + "Cs_log" + "/with/key/bPjDX0mOvggkBC15P93cy8"
        response = requests.post(url, data=payload)

# ここからスタート
def google_sheet():
        #print ("IFTTT連携開始")

        # IFTTT_Webhook
        flag=1
        if flag==1:
            print("蜜です!")
            print ("IFTTT連携終了")
        else:
            print("蜜は確認されませんでした")
