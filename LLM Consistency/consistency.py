from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def calculate_similarity(text_samples, tokenizer, model):
    embeddings = [get_embedding(text, tokenizer, model).squeeze().numpy() for text in text_samples]
    
    n = len(embeddings)
    similarity_scores = []
    for i in range(n):
        for j in range(i+1, n):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            similarity_scores.append(sim)
    
    average_similarity = sum(similarity_scores) / len(similarity_scores)
    return average_similarity

# Example usage
text_samples = [
    "Akila Dananjaya is a Sri Lankan cricketer who plays for the Sri Lankan national team. A right-handed batsman and right-arm off break bowler, he has taken 44 international wickets in 22 Twenty20 Internationals and 25 wickets in 16 One Day Internationals at an average of 26.68 since his debut on 10 November 2017 versus Pakistan.\n\nDananjaya made his List A cricket debut against the Indian Board President's XI on 12 February 2013. He was selected the following year for the Bangladesh tour in January 2014, and for the series against England in November 2014.\n\nDananjaya took five wickets against Zimbabwe in four One Day Internationals in the 2017-18 season, and was eventually included in the national squad as a back-up to Seekkuge Prasanna during the 2018 Asia Cup. He made his debut in December 2017, taking 5/50 in an innings against India in the second Test match of the same series. He made his ODI debut against Pakistan in October 2018, taking 4/30 in the first ODI. \n\nDan",
    "Akila Dananjaya is a Sri Lankan cricketer who plays for the Sri Lankan national cricket team and for the Tamil Union Cricket Club. He is a right-arm offbreak bowler, who can also bowl legspin. Dananjaya made his international debut in 2017 and took 5/24 in his maiden ODI in the same year. He has been described as having a 'mysterious repertoire of deliveries', and has the ability to deceive a batsman with his variations. He was named in the Sri Lankan squad for the 2019 Cricket World Cup and he picked up 6 wickets in the 2 Group stage matches and a 5-for in the semi final, which made him one of the most successful bowlers in the tournament. His performance in the World Cup and his ability to bowl effectively at the death earned him a place in the ICC ODI Team of the year 2019.",
    "Akila Dananjaya is a professional Sri Lankan cricketer. He is an off-break bowler who also bats right-handed and represents Sri Lanka in all forms of the game.\n\nDananjaya made his first-class debut for Colts Cricket Club in the 2012\u201313 season of the Premier Trophy. He made his Test debut for Sri Lanka against South Africa in January 2018, registering figures of 5/24 in the second innings. He achieved a career-best 6/46 in his fifteenth Test match against Bangladesh in February 2020.\n\nDananjaya has played in a number of Twenty20 (T20) cricket tournaments. He made his T20I debut for Sri Lanka against Zimbabwe in October 2016, taking three wickets in the opening match. He made his IPL debut for Delhi Capitals, also known as Delhi Daredevils, in the 2018 season. Dananjaya won the Emerging Player of the Month award in the 2021 Indian Premier League.",
    "Derek King is a former Australian rules football player and assistant coach of the Sydney Swans in the Australian Football League (AFL). He played for the Melbourne Football Club in the 1980s and the Fremantle Football Club in the 1990s.\n\nKing was originally from Melbourne and played for his local junior team South Melbourne. He was drafted by the Demons at pick 33 in the 1985 VFL draft and went on to play 60 AFL games for Melbourne over six seasons. He also made regular appearances in the VFL during his time with the Demons.\n\nDuring his time at Melbourne, King was an undersized forward who made his presence known despite his small stature. His bravery and willingness to back into a contest saw him become a vital asset to the team in the forward pocket.\n\nKing made the move to Fremantle in 1991 and would later go on to play a further 66 games for the Dockers, many of them as captain. During his time at Fremantle, King was known for his leadership and strong sense of team spirit, making him a popular figure among his teammates and coaches.",
    "Derek King (born 5 February 1951) is an Australian former professional footballer. He played for clubs such as Adelaide City, West Adelaide and Inter Milan during his career. King began his career with Adelaide City in 1969 and stayed until 1971, when he signed with Italian side Inter Milan. He quickly made an impact at Inter and scored one of the most memorable goals of his career when he struck a free-kick against Foggia in the Coppa Italia. He had a brief stint in Serie A before returning to Australia, where he signed with West Adelaide in 1974. King won two National Soccer League titles with Adelaide City in 1972 and 1974. King retired in 1979 and was inducted into the FFA Hall of Fame in 2006.",
]

average_similarity = calculate_similarity(text_samples, tokenizer, model)
print(f"Average similarity: {average_similarity}")
