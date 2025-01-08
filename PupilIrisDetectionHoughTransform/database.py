
import sqlite3
import fileUtil


def connect():
    conn = sqlite3.connect('example.db')
    c = conn.cursor()
    return (c, conn)
def commitClose(conn):
    conn.commit()
    conn.close()    

def getAllData(table):
    (c, conn) = connect()
    ind=0;
    for row in conn.execute("select * from "+table):
        ind+=1
        fileUtil.FileSave("backupKasaPupil.txt",str(row)+'\n')
        print(ind,": ",row)
        #c.execute("INSERT INTO 'pupilSegmentationResults' VALUES (?, ?, ?, ?, ?, ?, ?)", (row[0], row[1], row[2],row[3],row[4],row[5],row[6]));
    conn.close()
    return ind

def getBestRecord(table,fileName, folder, database,estimation):
    (c, conn) = connect()
    for row in c.execute("SELECT * FROM "+table+" where fileName=? and folder=? and database=? and estimation=?",(fileName, folder, database,estimation)):
        conn.close()
        return row
def getRecord(table,fileName, folder, database,method):
    (c, conn) = connect()
    for row in c.execute("SELECT * FROM "+table+" where fileName=? and folder=? and database=? and method=?",(fileName, folder, database,method)):
        conn.close()
        return row


def ifExistRecord(table,fileName, folder, database,method):
    (c, conn) = connect()
    c.execute("SELECT * FROM "+table+" where fileName=? and folder=? and database=? and method=?",(fileName, folder, database,method))
    row = c.fetchone()
    return row  
    
def insertData(table, fileName, folder, database, center, radius, estimation,method):
    center= str(center)
    radius = int(radius)
    (c, conn) = connect()
    c.execute("INSERT OR REPLACE INTO "+table+" VALUES (?, ?, ?, ?, ?, ?, ?)", (fileName, folder, database, center, radius, estimation,method))
    commitClose(conn)

def renameTable(fromTable,toTable):
    (c, conn) = connect()
    c.execute('''ALTER TABLE '''+fromTable+''' RENAME TO '''+ toTable)
    commitClose(conn)
def addColumnMethod(table):
    (c, conn) = connect()
    c.execute('''ALTER TABLE '''+table+''' ADD method VARCHAR(50) NOT NULL  DEFAULT 'kasa' ''')
    commitClose(conn)

def createTable(table):
    (c, conn) = connect()
    c.execute('''CREATE TABLE if not exists '''+table+'''
                (fileName text, folder text, database text, center text, radius int, estimation text, method text, PRIMARY KEY (fileName, folder, database,method))''')
    commitClose(conn)

def dropAllTables():
    (c, conn) = connect()
    c.execute('''drop table if exists kasaPupil''')
    commitClose(conn)
    
