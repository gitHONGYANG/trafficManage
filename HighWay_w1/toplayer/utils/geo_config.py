'''
地理信息
监控路段从TV36-TV63																																							入口
																																											 \																																	                                     \
___TV36_______________TV37_______________TV38__________________TV39____________________TV40______________________TV41_____________________TV42____________________TV43_______\_________________TV44_____________________
																																													\
																																													\
																																													出口
																																			服务区
																																			   \
____TV45___________________TV46______________________TV47_____________________TV48____________________TV49_______________________TV50__________\___________TV51______________________TV52________________________TV53_______________________TV54_____________



									   入口																			 入口
										\																			   \
____TV55___________________TV56_________\______________TV57______________________TV58____________________TV59__________\____________TV60_______________________TV61__________________________TV62_______________________________TV63_________________________________________________________
				\																											\
				\																											\
			   出口																										   出口
'''
#pix max 1366
from pymongo import MongoClient
url = 'localhost:27017'
dbname = 'highway'
config_tb = MongoClient(url)[dbname]['geoconfig']

'''
tv36 = {'name':'TV36', 'dk': 898, 'position':500}
tv37 = {'name':'TV37', 'dk': 898, 'position':500}
tv38 = {'name':'TV38', 'dk': 898, 'position':500}
tv39 = {'name':'TV39', 'dk': 898, 'position':500}
tv40 = {'name':'TV40', 'dk': 898, 'position':500}
tv41 = {'name':'TV41', 'dk': 898, 'position':500}
tv42 = {'name':'TV42', 'dk': 898, 'position':500}
tv43 = {'name':'TV43', 'dk': 898, 'position':500}
tv44 = {'name':'TV44', 'dk': 898, 'position':500}
tv45 = {'name':'TV45', 'dk': 898, 'position':500}
tv46 = {'name':'TV46', 'dk': 898, 'position':500}
tv47 = {'name':'TV47', 'dk': 898, 'position':500}
tv48 = {'name':'TV48', 'dk': 898, 'position':500}
tv49 = {'name':'TV49', 'dk': 898, 'position':500}
tv50 = {'name':'TV50', 'dk': 898, 'position':500}
tv51 = {'name':'TV51', 'dk': 898, 'position':500}
tv52 = {'name':'TV52', 'dk': 898, 'position':500}
tv53 = {'name':'TV53', 'dk': 898, 'position':500}
tv54 = {'name':'TV54', 'dk': 898, 'position':500}
tv55 = {'name':'TV55', 'dk': 898, 'position':500}
tv56 = {'name':'TV56', 'dk': 898, 'position':500}
tv57 = {'name':'TV57', 'dk': 898, 'position':500}
tv58 = {'name':'TV58', 'dk': 898, 'position':500}
tv59 = {'name':'TV59', 'dk': 898, 'position':500}
tv60 = {'name':'TV60', 'dk': 898, 'position':500}
tv61 = {'name':'TV61', 'dk': 898, 'position':500}
tv62 = {'name':'TV62', 'dk': 898, 'position':500}
tv63 = {'name':'TV63', 'dk': 898, 'position':500}
'''

if __name__ == '__main__':
	#config_tb.remove({})
	#lis = [tv36,tv37,tv38,tv39,tv40,tv41,tv42,tv43,tv44,tv45,tv46,tv47,tv48,tv49,tv50,tv51,tv52,tv53,tv54,tv55,tv56,tv57,tv58,tv59,tv60,tv61,tv62,tv63]
	#config_tb.insert_many(lis)
	print('done')
