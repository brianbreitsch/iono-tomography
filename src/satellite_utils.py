import urllib2
import urllib
from cookielib import CookieJar

import sys 
import argparse
import string

def convert_line_endings(temp, mode):
        """
        modes:  0 - Unix, 1 - Mac, 2 - DOS
        """
        if mode == 0:
                temp = string.replace(temp, '\r\n', '\n')
                temp = string.replace(temp, '\r', '\n')
        elif mode == 1:
                temp = string.replace(temp, '\r\n', '\r')
                temp = string.replace(temp, '\n', '\r')
        elif mode == 2:
                import re
                temp = re.sub("\r(?!\n)|(?<!\r)\n", "\r\n", temp)
        return temp

def fetch_tle(username, password):
    # list of gps satellites:
    # http://www.n2yo.com/satellites/?c=20
    basepath = 'https://www.space-track.org'
    auth_path = '/auth/login'
    query = '/basicspacedata/query/class/tle_latest/ORDINAL/NAVSTAR/orderby/ORDINAL asc/format/3le/metadata/false'
    #query = '/basicspacedata/query/class/tle_latest/OBJECT_NAME/NAVSTAR\ 1/orderby/ORDINAL\ asc/limit/5/format/3le/metadata/false'
    query = '/basicspacedata/query/class/satcat/LAUNCH/>now-7/CURRENT/Y/orderby/LAUNCH\ DESC/format/html'
    query = '/basicspacedata/query/class/tle/NORAD_CAT_ID/39166/orderby/EPOCH asc/limit/5/format/3le/metadata/false'
    query = '/basicspacedata/query/class/tle_latest/NORAD_CAT_ID/39166/orderby/ORDINAL asc/limit/3/metadata/false'
    #query = '/basicspacedata/query/class/tle_latest/ORDINAL/1/favorites/COSMIC/format/3le/emptyresult/show'

    cj = CookieJar()
    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))

    login_str = 'identity=' + username + '&password=' + password
    response = opener.open(basepath + auth_path, login_str)
    content = response.read()

    response = opener.open(basepath + query)
    content = response.read()
    return content



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change line endings.')
    parser.add_argument('--unix', action='store_true')
    parser.add_argument('--macos9', action='store_true')
    parser.add_argument('--dos', action='store_true')
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)

    args = parser.parse_args()

    mode = 0 
    if args.macos9: mode = 1 
    elif args.dos: mode = 2 

    for line in args.infile:
        args.outfile.write(convert_line_endings(line, mode))


