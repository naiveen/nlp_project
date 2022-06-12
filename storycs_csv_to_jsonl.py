import pandas
import json
import regex as re


df = pandas.read_csv("data/storycs/pro_data_test.csv",)

file_to_write = ""
# print (df.columns)
result=df.to_json(orient='records')
emotions = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

result_list=json.loads(result)
entries_to_remove=['storyid','char_maslow','char_reiss','char_plutchik']

def get_emotion_label(emotion_votes):
    max_emo=''
    if(max(emotion_votes)>0):
        max_emo=emotions[emotion_votes.index(max(emotion_votes))]
    return max_emo


with open("data/storycs/test.jsonl","w") as outfile:
    for example in result_list:
        labels=[]
        chars_list=example['char'].strip('][').split(',')
        example['char_plutchik']=re.sub("{'",'{"',example['char_plutchik'])
        example['char_plutchik']=re.sub("':",'":',example['char_plutchik'])

        # print(example['char_plutchik'])
        char_vote_list=json.loads(example['char_plutchik'])
        example['labels']=[]
        for char_votes in char_vote_list:
            # print(char_votes)
            for char in char_votes:
                emotion_votes=char_votes[char]
                emotion_label=get_emotion_label(emotion_votes)
                if(emotion_label==''): continue
                example['labels'].append({char:emotion_label})
        if len(example['labels'])==0:
            continue
        for entry in entries_to_remove:
            example.pop(entry)


        outfile.write(json.dumps(example)+'\n')