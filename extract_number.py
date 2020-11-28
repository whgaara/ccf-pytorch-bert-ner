import re


def extract_mobile(text):
    mobile_list = re.findall(r"1\d{10}|\d{2,3}-\d{7,8}|\d{3}-\d{3,4}-\d{4}|\+\d{12}|[\(\（]\d{2,4}[\)\）]\d{7,8}|\+\d{2}-\d{11}|[\(\（]\d{2,3}[\)\）]\d{4}-\d{4}-\d{3}", text)
    #匹配11位手机号 和座机类型 : '400-692-0001' 与 010-55667890' 与'+886223755010' 与'（852）94896744' 与'+86-18601200718' 与（86）1380-2841-004
    mobile_list = list(set(mobile_list)) #去重 去掉邮箱中的手机号
    mobile_res_all = []
    for mobile in mobile_list:
        mobile_res_single = ['mobile']
        pos = re.search(re.escape(mobile), text).span() #提取position re.escape转义字符()（）+_
        mobile_res_single.extend([pos[0], pos[1]-1])
        mobile_res_single.append(mobile)
        #print(mobile_res_sigle)
        mobile_res_all.append(mobile_res_single)
    return mobile_res_all

def extract_qq(text):
    qq_list = re.findall(r"(QQ|QQ[:：]|Q群[:：])([1-9]\d{4,10})", text)
    qq_list = list(set(qq_list)) #去重 去掉qq邮箱中的qq号
    qq_res_all = []
    for qq in qq_list:
        qq_res_single = ['QQ']
        qq_number = qq[1] #提取出纯数字
        pos = re.search(qq_number, text).span() #提取position
        qq_res_single.extend([pos[0], pos[1]-1])
        qq_res_single.append(qq_number)
        #print(qq_res_single)
        qq_res_all.append(qq_res_single)
    return qq_res_all

def extract_vx(text):
    vx_list = re.findall(r"(微信[：:]|微信|微信号|微信号[：:])([A-Za-z0-9\-\_]{6,20})", text)
    vx_res_all = []
    for vx in vx_list:
        vx_res_single = ['vx']
        vx_number = vx[1] #  提取微信号内容 '('微信号：', 'liushasha319439')-> 'liushasha319439'
        pos = re.search(vx_number, text).span() #提取position
        vx_res_single.extend([pos[0], pos[1]-1])
        vx_res_single.append(vx_number)
        #print(vx_res_single)
        vx_res_all.append(vx_res_single)
    return vx_res_all


def get_emailAddress(text):
    # emailRegex = r'^[0-9a-zA-Z_]{0,19}@[0-9a-zA-Z]{1,13}\.[com,cn,net]{1,3}$'
    emailRegex = r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+'
    email = re.search(emailRegex, text)
    # print(email)
    if email:
        start = email.start()
        end = email.end()
        group = email.group()
        # print(['email', start, end, group])
        return ['email', start, end, group]
    else:
        return None
