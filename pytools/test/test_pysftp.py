import pysftp

cnopts = pysftp.CnOpts()
cnopts.hostkeys = None

dir_path = '/data2/home/zhouxiangyong/Workspace/Dev/AortaSlice/data/aorta_extract_data1_fxc'

with pysftp.Connection('s108', port=10822, username='zhouxiangyong', password='zxy201811', cnopts=cnopts) as sftp:
    # out = sftp.execute('cd {}; ls'.format(dir_path))
    sftp.cwd(dir_path)
    out = sftp.listdir()
    print(out)
