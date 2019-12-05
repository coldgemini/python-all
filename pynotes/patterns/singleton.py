class CfgSingleton:
    class __Config:
        def __init__(self):
            self.cfg = None
            self.name = 'NONAME'

        def read_cfg(self, cfg):
            self.cfg = cfg
            self.name = cfg['NAME']

        def __str__(self):
            return repr(self) + self.cfg

    instance = None

    def __init__(self):
        if not CfgSingleton.instance:
            CfgSingleton.instance = CfgSingleton.__Config()
        # else:
        #     raise KeyError('config already exists')

    def read_cfg(self, cfg):
        self.instance.read_cfg(cfg)

    def __getattr__(self, name):
        return getattr(self.instance, name)

    # def __getitem__(self, key):
    #     return self.cfg.get(key, None)


if __name__ == '__main__':
    cfg_0 = {'NAME': 'model1'}
    cfg_singleton = CfgSingleton()
    print(cfg_singleton.name)
    # print(cfg_singleton['NAME'])
    cfg_singleton.read_cfg(cfg_0)
    print(cfg_singleton.name)

    cfg_singleton_1 = CfgSingleton()
    print(cfg_singleton_1.name)
