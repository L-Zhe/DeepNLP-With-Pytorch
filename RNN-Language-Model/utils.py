from    math import ceil
import  sys

def view_bar(num, total, epoch, EPOCH):
    rate = num / total
    rate_num = int(ceil(rate * 100))
    r = '\r[%d/%d][%s%s]%d%% ' % (epoch, EPOCH, "=" * rate_num, " " * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()