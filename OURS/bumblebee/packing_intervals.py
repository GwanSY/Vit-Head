"""Exact integer-nanosecond interval intersections for large native traces.

Only offline bookkeeping is optimized. The shared right-hand event list is
immutable within each party's audit; cache its union once and use prefix sums.
"""
from bisect import bisect_right


def merged(rows):
    result=[]
    for left,right in sorted((r['start_ns'],r['end_ns']) for r in rows):
        if result and left<=result[-1][1]:result[-1][1]=max(right,result[-1][1])
        else:result.append([left,right])
    return result


class IndexedIntersection:
    def __init__(self):
        self.right=None

    def __call__(self,a,b):
        if self.right is not b:
            self.right=b
            spans=merged(b)
            self.starts=[x[0] for x in spans]
            self.ends=[x[1] for x in spans]
            self.prefix=[0]
            for left,right in spans:self.prefix.append(self.prefix[-1]+right-left)
        def coverage(t):
            j=bisect_right(self.ends,t)
            return self.prefix[j]+(max(0,t-self.starts[j]) if j<len(self.starts) else 0)
        return sum(coverage(right)-coverage(left) for left,right in merged(a))/1e9


def check():
    import csv
    import random
    from pathlib import Path
    from cost_analysis import intersection_seconds
    rng=random.Random(20261016);fast=IndexedIntersection();count=0
    for offset in (0,1791990311527913278):
        for n in (0,1,2,10,100):
            b=[dict(start_ns=offset+(s:=rng.randrange(1000)),end_ns=offset+s+rng.randrange(0,100)) for _ in range(n)]
            for m in (0,1,2,30):
                a=[dict(start_ns=offset+(s:=rng.randrange(1000)),end_ns=offset+s+rng.randrange(0,100)) for _ in range(m)]
                assert fast(a,b)==intersection_seconds(a,b)
                count+=1
    # Cross-check real large, overlapping multi-thread link intervals, not
    # merely synthetic cases. All comparisons are exact to integer nanoseconds.
    sample=Path('out/14/full/log/full-r01-S10R2-baseline/log/native-p0-e1.tsv')
    if sample.exists():
        events=[]
        with sample.open() as f:
            for r in csv.DictReader(f,delimiter='\t'):
                events.append(dict(kind=r['kind'],name=r['name'],start_ns=int(r['start_ns']),end_ns=int(r['end_ns'])))
        links=[r for r in events if r['kind'] in ('send','recv')]
        windows=[r for r in events if r['kind']=='HLO'][::400]
        for window in windows:
            assert fast([window],links)==intersection_seconds([window],links)
            count+=1
    print('Exact interval comparisons passed:',count)


if __name__=='__main__':check()
