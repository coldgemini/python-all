import json
f=open('stus.json',encoding='utf-8')
user_dic=json.load(f)
print(user_dic)


stus={'xiaojun':'123456','xiaohei':'7890','lrx':'111111'}
f=open('stus2.json','w',encoding='utf-8')
json.dump(stus,f,indent=4,ensure_ascii=False)