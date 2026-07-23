import pandas as pd

data = {
    'userName': [
        'User_001_MovieFan',
        'User_002_AngryBird',
        'User_003_ChillGuy',
        'User_004_Disappointed',
        'User_005_HappyCamper',
        'User_006_TechGuru',
        'User_007_NeutralObserver',
        'User_008_CouchPotato',
        'User_009_BudgetWatcher',
        'User_010_LoyalSub'
    ],
    'content': [
        "I absolutely love the new features. Netflix is getting better every day and the streaming quality is excellent!",
        "The worst experience ever. The app keeps crashing, it's terrible, and it's way too expensive now.",
        "It's okay, nothing special but I enjoy the variety of movies. It passes the time.",
        "Terrible customer service. I hate the new update so much, it is completely useless.",
        "Great recommendations! I can always find something good to watch on a Friday night.",
        "I can't even open the app anymore. Very bad experience, considering canceling my subscription.",
        "Good app, but sometimes the subtitles are out of sync. Needs some minor improvements.",
        "Excellent streaming quality, no buffering at all. Love the new original series!",
        "Why is the subscription price increasing? The content is bad now and not worth it anymore.",
        "A decent app overall. Has some bugs but they get fixed eventually. Still the best streaming platform."
    ]
}

df = pd.DataFrame(data)
df.to_excel('sample_10_organic.xlsx', index=False)
print("Berhasil membuat sample_10_organic.xlsx")
