'''
input: point, tv_name
output: 1(at left); 0(at right)
'''
import numpy as np

def region_judge(point, tv_name):
    judge_dict = {
                  'tv3': ((817, 303), (1100, 1080)),
                  'tv4': ((859, 421), (1066, 983)),
                  'tv5': ((892, 498), (1115, 992)),
                  'tv6': ((987, 454), (1201, 983)),
                  'tv7': ((814, 463), (1025, 968)),
                  'tv8': ((1031, 479), (1230, 980)),
                  'tv9': ((877, 467), (1074, 983)),
                  'tv10': ((819, 378), (1035, 983)),
                  'tv11': ((864, 416), (1015, 797)),
                  'tv12': ((839, 311), (1139, 1073)),
                  'tv13': ((928, 454), (1140, 981)),
                  'tv14': ((719, 357), (943, 962)),
                  'tv15': ((894, 424), (1094, 985)),
                  'tv16': ((914, 493), (1116, 1009)),
                  'tv17': ((882, 475), (1053, 983)),
                  'tv18': ((833, 281), (1141, 1075)),
                  'tv19': ((846, 356), (1071, 969)),
                  'tv20': ((889, 425), (1100, 966)),
                  'tv21': ((926, 409), (1170, 985)),
                  'tv22': ((861, 421), (1043, 985)),
                  'tv23': ((942, 441), (1163, 977)),
                  'tv24': ((899, 458), (1106, 976)),
                  'tv25': ((1156, 661), (1258, 986)),
                  'tv26': ((1006, 439), (1192, 989)),
                  'tv27': ((764, 343), (896, 749)),
                  'tv28': ((919, 431), (1095, 1038)),
                  'tv29': ((1059, 427), (1297, 1053)),
                  'tv30': ((857, 314), (1031, 819)),
                  'tv31': ((942, 392), (1139, 1045)),
                  'tv32': ((790, 260), (945, 742)),
                  'tv33': ((889, 419), (1110, 1043)),
                  'tv34': ((790, 391), (1048, 1067)),
                  'tv35': ((815, 262), (1092, 1080)),
                  'tv36': ((943, 310), (1286, 1080)),
                  'tv37': ((849, 290), (1133, 1080)),
                  'tv38': ((994, 370), (1321, 1080)),
                  'tv39': ((873, 385), (1110, 1080)),
                  'tv40': ((807, 385), (1098, 1080)),
                  'tv41': ((897, 375), (1195, 1080)),
                  'tv42': ((904, 410), (1116, 1080)),
                  'tv43': ((937, 379), (1167, 1080)),
                  'tv44': ((831, 375), (1070, 1080)),
                  'tv45': ((948, 472), (1198, 1080)),
                  'tv46': ((922, 498), (1072, 1080)),
                  'tv47': ((957, 419), (1222, 1080)),
                  'tv48': ((1023, 427), (1211, 1080)),
                  'tv49': ((1118, 507), (1280, 1080)),
                  'tv50': ((736, 412), (1010, 1080)),
                  'tv51': ((849, 290), (1076, 1080)),
                  'tv52': ((838, 405), (1045, 1080)),
                  'tv53': ((964, 370), (1090, 1080)),
                  'tv54': ((869, 480), (1118, 1080)),
                  'tv55': ((1025, 509), (1333, 1080)),
                  'tv56': ((857, 474), (1116, 1080)),
                  'tv57': ((904, 487), (1160, 1080)),
                  'tv58': ((855, 507), (1096, 1080)),
                  'tv59': ((862, 352), (1092, 1080)),
                  'tv60': ((906, 494), (1142, 1080)),
                  'tv61': ((961, 480), (1171, 1080)),
                  'tv62': ((917, 445), (1240, 1080)),
                  'tv63': ((944, 441), (1165, 1080)),
                  'tv64': ((860, 421), (1083, 1080)),
                  'tv65': ((599, 226), (836, 822)),
                  'tv66': ((748, 494), (936, 1060)),
                  'tv67': ((800, 390), (1109, 1063)),
                  'tv68': ((747, 407), (1092, 1066)),
                  'tv69': ((722, 457), (973, 1061)),
                  'tv70': ((766, 494), (1037, 1074)),
                  'tv71': ((850, 409), (1034, 1066)),
                  'tv72': ((754, 504), (929, 1045)),
                  'tv73': ((805, 474), (1033, 1074)),
                  'tv74': ((863, 404), (1130, 1070)),
                  'tv75': ((1083, 593), (1307, 1055)),
                  'tv76': ((800, 607), (1010, 1066)),
                  'tv77': ((888, 448), (1077, 969)),
                  'tv78': ((942, 487), (1202, 1060)),
                  'tv79': ((936, 445), (1157, 1064)),
                  'tv80': ((867, 486), (1067, 1063)),
                  'tv81': ((764, 467), (1011, 1069)),
                  'tv82': ((900, 476), (1073, 1069)),
                  'tv83': ((899, 377), (1242, 1068)),
                  'tv84': ((877, 461), (1022, 820)),
                  'tv85': ((821, 497), (1019, 1061)),
                  'tv86': ((1020, 512), (1205, 1067))                  
}
    (x, y) = (point[0], point[1])
    (x1, y1) = judge_dict[tv_name][0]
    (x2, y2) = judge_dict[tv_name][1]
    left_flag = (y - y2) * (x2 - x1) - (x - x2) * (y2 - y1)
    if left_flag > 0:
        #print('left')
        return 'edge'
    else:
        #print('right')
        return 'road'