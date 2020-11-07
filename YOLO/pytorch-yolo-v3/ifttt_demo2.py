import requests

# IFTTT_Webhook
def ifttt_webhook(eventid):
    payload = {"value1": "蜜",
                "value2": "じゃないよ",
                "value3": "♡" }
    url = "https://maker.ifttt.com/trigger/" + "google_home" + "/with/key/bPjDX0mOvggkBC15P93cy8"
    response = requests.post(url, data=payload)

# ここからスタート
def googleDense():
    print ("IFTTT連携開始")

    # IFTTT_Webhook
    flag=1
    if flag==1:
        ifttt_webhook("line_event")
        print("蜜です!")
        print ("IFTTT連携終了")
    else:
        print("蜜は確認されませんでした")
