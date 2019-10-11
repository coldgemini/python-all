import yaml

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

print(type(cfg))
for section in cfg.keys():
    print(type(section))
    print(section)
print(cfg['mysql'])
print(cfg['other'])

print(cfg['mysql']['user'])

print(cfg['env'])
print(cfg['env']['LEARNING_RATE'])
print(type(cfg['env']['LEARNING_RATE']))
print(cfg['env']['BATCH_SIZE'])
print(type(cfg['env']['BATCH_SIZE']))
print(cfg['env']['SPLIT'])
print(cfg['data']['boundaries'])
