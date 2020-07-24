from modules.print_manager import PrintManager
import sys

file_major_name = 'file_major.log'
file_minor_name = 'file_minor.log'
file_major = open(file_major_name, 'w')
file_minor = open(file_minor_name, 'w')
print_manager_major = PrintManager()


#print(sys.stdout)
print_manager_major.file_printing(file_major)
print('major_only')
#print(sys.stdout)
print_manager_minor = PrintManager()
print_manager_minor.file_printing(file_minor)
print('major_and_minor')
#print(sys.stdout)
print_manager_minor.close()
file_minor.close()
print('major_only_2')
#print(sys.stdout)
print_manager_major.close()
file_major.close()
