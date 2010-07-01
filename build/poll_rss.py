import time
import commands

#simple script to track compiler memory usage on a Linux system
# execute with $ python poll_rss.py and use Ctrl-C to exit

def poll_rss(interval=0.1):
    try:
        wait = 0.1
        
        cmds = {}
        
        while True:
            output = commands.getoutput('ps -eo rss,cmd | grep -E "cc1plus|ptxas|nvcc" | grep -vE "grep"')
        
            if output == '': continue
        
            for line in output.split('\n'):
                tokens = line.strip().split(' ')
                rss = int(tokens[0])
                cmd = ' '.join(tokens[1:])
        
                if cmd in cmds:
                    cmds[cmd] = max(cmds[cmd], rss)
                else:
                    cmds[cmd] = rss
        
            time.sleep(interval)
    except:
        cmds = [(v,k) for (k,v) in cmds.items()]
        cmds.sort(reverse=True)
        for (rss,cmd) in cmds:
            if len(cmd) > 200:
                cmd = cmd[:100] + ' ... ' + cmd[-100:]
            print "%10.1f MB  |  %s" % (rss/1024.0,cmd)

if __name__ == '__main__':
    print "Polling resident set size of compiler chain... [Ctrl-C to terminate]"
    poll_rss()

