import pandas
import json
import regex as re
import numpy as np
df = pandas.read_csv("data/storycs/pro_data_dev.csv",)

file_to_write = ""
# print (df.columns)
dev_range= list(np.random.choice(list(range(14637)),size=2000,replace=False))
test_range= list(np.random.choice(list(range(14216)),size=2000,replace=False))
dev_range_dict= {k:True for k in dev_range}
test_range_dict= {k:True for k in test_range}

range_dict=dev_range_dict

result=df.to_json(orient='records')
emotions = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

result_list=json.loads(result)
entries_to_remove=['storyid','char_maslow','char_reiss','char_plutchik']

def get_emotion_label(emotion_votes):
    max_emo=''
    if(max(emotion_votes)>0):
        max_emo=emotions[emotion_votes.index(max(emotion_votes))]
    return max_emo

idx=0
with open("data/storycs/dev2k.jsonl","w") as outfile:
    for example in result_list:
        labels=[]
        chars_list=example['char'].strip('][').split(',')
        example['char_plutchik']=re.sub("{'",'{"',example['char_plutchik'])
        example['char_plutchik']=re.sub("':",'":',example['char_plutchik'])

        # print(example['char_plutchik'])
        char_vote_list=json.loads(example['char_plutchik'])
        for entry in entries_to_remove:
            example.pop(entry)

        for char_votes in char_vote_list:
            # print(char_votes)
            for char in char_votes:
                emotion_votes=char_votes[char]
                emotion_label=get_emotion_label(emotion_votes)
                if(emotion_label==''): continue
                example['label']=emotion_label
                example['person']=char
                if(idx in range_dict):
                    outfile.write(json.dumps(example)+'\n')   
                idx+=1

