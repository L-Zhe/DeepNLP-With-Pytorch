import sys
import time


def view_bar(num, total, epoch):
    rate = num / total
    rate_num = int(rate * 100)
    r = '\r[%d][%s%s]%d%% ' % (epoch,"="*rate_num, " "*(100-rate_num), rate_num, )

    sys.stdout.write(r)
    sys.stdout.flush()

print(1, end='')

for i in range(200):
    time.sleep(0.01)

    view_bar(i, 199, 233)

