## Input
 - "Brot wird aus Mehl gebacken"
 - "Aus was ist Brot gemacht?"
 - "Im Straßenverkehr fahren viele Autos"
 - "Mit dem Auto kann man zum Bäcker fahren"

## Ziel

    Für gegebenen Input herausfinden welche anderen Inputs am ähnlichsten sind

 - mehl   [1,1,0,0,0]
 - brot   [0,1,0,0,0]
 - bäcker [0,1,1,0,0]
 - 
 - fahren [0,0.1,0.1,1,0]
 - auto   [0,0,0,1,0]
 - straßenverkehr

## Beispiel

 - mehl   [1,1,0,0,0]
 - brot   [0,1,0,0,0]
 - bäcker [0,1,1,0,0]

"mehl brot"
-> [0.5,1,0,0,0]

 ### x:

  - [1,1,0,0,0]

### y:

  - [0,1,0,0,0]

| X           | Y  |
|:-------------:| -----:|
|  mehl | brot |
| brot      |   mehl |
| mehl      |    bäcker |
| brot      |    bäcker |
| bäcker      |    brot |
| bäcker      |    fahren |
| fahren      |    auto |

brot: [0,1,0,0,0,0]
weizen: [0,0,0,0,0,1]

Brot-> vector

Brot: [0,0.95,0,0,0,0.05]
20x Brot


Bäcker
5x Brot

Weizen
1x Brot