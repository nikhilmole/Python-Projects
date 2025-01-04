import os
import sys
import hashlib
import time
import schedule
import smtplib
import psutil
from sys import *
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import urllib.request as urllib2

def is_connected():
    try:
        urllib2.urlopen('http://www.google.com', timeout=5)
        return True
    except urllib2.URLError as err:
        print("URL Error:",err)
        return False

def MailSender(filename,time):
    try:
        fromaddr = "nikhilmole3@gmail.com"
        toaddr = "picapica2698@gmail.com"

        msg = MIMEMultipart()

        msg['From'] = fromaddr
        msg['To'] = toaddr

        body = """
        Hello %s,
        Welcome to Marvellous infosystem.
        Please fined Attached Document which contauin Log of Running process.
        Log file is created at : %s

        This is auto generated mail

        Thanks and Regards,
        Piyush manohar kahirnar
        marvellous Infosystem
            """%(toaddr,time)
        
        subject = "Marvellous Infosystem Process Log generated at: "+time


        msg['Subject'] = subject
        msg.attach(MIMEText(body,'plain'))

        attachment = open(filename,"rb")

        p = MIMEBase('application','octet-stream')

        p.set_payload((attachment).read())

        encoders.encode_base64(p)

        p.add_header('Content-Disposition',"Attachment; filename = %s" %filename)

        msg.attach(p)

        s = smtplib.SMTP('smtp.gmail.com',587)

        s.starttls()

        s.login(fromaddr,"tdoyvxfygsuqqesf")

        text = msg.as_string()

        s.sendmail(fromaddr,toaddr,text)

        s.quit()

        print("log file successfully send through mail")

    except Exception as E:
        print("Unable to send mail.",E)

def ProcessLog(log_dir = 'Marvellous'):
    listprocess = []

    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except Exception as D:
            print("Errot as %s",D)
    
    separator = '-' * 80
    log_path = os.path.join(log_dir,"MarvellousLog%s.log"%(time.ctime()))
    log_path = log_path.replace(' ','_').replace(':','_')
    f = open(log_path,'w')
    f.write(separator +"\n")
    f.write("Marvellous Infosystem process logger: " +time.ctime() + "\n")
    f.write("\n")

    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=(['pid','name','username']))
            vms = proc.memory_info().vms / (1024 * 1024)
            pinfo['vms'] = vms
            listprocess.append(pinfo)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    for element in listprocess:
        f.write("%s\n" %element)

    print("Log file is successfully generated at location %s",(log_path))

    connected = is_connected()

    if connected:
        starttime = time.time()
        MailSender(log_path,time.ctime())
        endtime = time.time()

        print('Took %s seconds to send mail' %(endtime - starttime))

    else:
        print("There is no internet connection")

def main():
    print("----Marvellous Infosystem by piyush khairnar----")

    print("Application name :" +argv[0])

    if(len(sys.argv) != 2):
        print("Error: Invalid number of arguments")
        exit()

    if(sys.argv[1] == "--h") or (sys.argv[1] == "--H"):
        print("This script is used log record of running processess amd send through mail")
        exit()
    
    if(sys.argv[1] == "--u") or (sys.argv[1] == "--U"):
        print("Usage: Applicationnmae AbsolutePath_of_Directory")
        exit()

    try:
        schedule.every(int(argv[1])).minutes.do(ProcessLog)
        while True:
            schedule.run_pending()
            time.sleep(1)

    except ValueError:
        print("Error: Invalid datatype of input")

    except Exception as E:
        print("Error: Invalid input",E)

if __name__ == "__main__":
    main()