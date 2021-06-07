from abc import ABC, abstractmethod
import numpy as np
from rates import Rates_Jurai
from common import f_nu

class Quadrature(ABC):

    def kc_list(self):
        pass

    @abstractmethod
    def weights(self):
        pass

    @abstractmethod
    def rates(self):
        pass

class TrapezoidalQuadrature(Quadrature):

    # Can use with either rates class
    def __init__(self, kc_list, rates_interface):
        self._kc_list = kc_list

        if (len(kc_list)) == 1:
            self._weights = np.array([1.0])
        else:
            self._weights = [0.5 * (kc_list[1] - kc_list[0])]

            for i in range(1, len(kc_list) - 1):
                self._weights.append(kc_list[i + 1] - kc_list[i])

            self._weights.append(0.5 * (kc_list[-1] - kc_list[-2]))

        self._rates = []

        for kc in self._kc_list:
            self._rates.append(rates_interface.get_rates(kc))

    def kc_list(self):
        return self._kc_list

    def weights(self):
        return self._weights

    def rates(self):
        return self._rates

class GaussFermiQuadrature(Quadrature):

    # Generated using "A Software Repository for Gaussian Quadratures and Christoffel Functions"
    # URL: https://epubs.siam.org/doi/book/10.1137/1.9781611976359?mobileUi=0
    _gf_weights = [
        [
            [1.1865691104156254528217229759472, 0.69314718055994530941723212145818]
        ],
        [
            [0.69053925540417512890181074851165, 0.57466334789701047736827168182564],
            [3.592384070408227709788009384086, 0.11848383266293483204896043963254]
        ],
        [
            [0.48405347515540606375507943615906, 0.46336055192044521369043419515716],
            [2.44674486896708526687511898042, 0.22074941357747184270152112188024],
            [6.4243522612255152565859012563254, 0.0090372150620282530252768044207816]
        ],
        [
            [0.37127566208502766166238611095501, 0.38253881569521505863866719132506],
            [1.8754328208332048777148706097806, 0.27630124176988759097334013356364],
            [4.6614710882608501705687026921922, 0.033825533446068277654868714321303],
            [9.5070936620765041150435024301794, 0.00048158964877438215035608224818596]
        ],
        [
            [0.30047710047452406690384490164606, 0.32380834646907132964826759989653],
            [1.5241528720170766786261022488047, 0.30048082347268964417797601237134],
            [3.7137045036461474203456894184864, 0.065591395786262153807404291533572],
            [7.1899575240030151108933728403272, 0.0032454517267055762175399767394786],
            [12.739543173222200611580048663786, 0.000021163105216605566044240917257187]

        ],
        [
            [0.25201744581969162719163621068446, 0.27987665796553372753543243022392],
            [1.2843141493663215106586844750499, 0.30683441743542631263672336936426],
            [3.1029981710598976943979230683672, 0.096829688278155764760188309372829],
            [5.8737898712073893451183019052874, 0.0093677513081998465676285341562393],
            [9.9296052965964677025588309880363, 0.00023784415324833480256766694843098],
            [16.072385345701815540163479769494, 0.00000082141938132311469181139249490017]
        ],
        [
            [0.21682333051723539200177134128145, 0.24601730415488904362165260034208],
            [1.1095932502568841443530922591579, 0.30358956987707971900315949869855],
            [2.6713656939695378277492869066634, 0.12396530227503054401693797519396],
            [4.9948394180545639177887622706795, 0.018577433359538532356055514333097],
            [8.2692851767021097267857678058229, 0.00098296100936000746859164147049949],
            [12.818248596365413313331263764553, 0.000014580696728489656468861715758412],
            [19.478268779804024339760293148533, 0.000000029187318973294366029704245242454]
        ],
        [
            [0.19013447473589591882320574794736, 0.21922665474944949865874883925958],
            [0.97647794762028115724709964141481, 0.29539006831933102574816170663505],
            [2.3480597400046232725974617925739, 0.14594722878843584580340278914327],
            [4.3577443121043584055150963007665, 0.029933731496840168111316703091028],
            [7.1291702199825029505841464323351, 0.0025649458974841044485248780521608],
            [10.838208319035682009373515852846, 0.000083765611049592340970604263832498],
            [15.818688054723173688626859634226, 0.00000078472712423372310951446476153719],
            [22.940145521136535495256277883271, 0.00000000097023084058299708654850371322567]
        ],
        [
            [0.16921763343494953069238790985473, 0.19755146694290501218265508363005],
            [0.87164484192232126939385169409868, 0.28481492721926227364490670599567],
            [2.0959199140448320082991643018344, 0.16294137436930644553601925672144],
            [3.8714445491399296824088140789902, 0.042403174441631869342963487694722],
            [6.2851600122707866296281037029193, 0.0051474372792987871576248349809181],
            [9.4491141430094516576278641612501, 0.00028264339287624812726914050286658],
            [13.540473650830151656596107752882, 0.0000061186823205661031086367181702782],
            [18.906778342620672283363059210026, 0.00000003820173826214879281825387109783],
            [26.44655514253360032941299114256, 0.000000000030605845173892156960467000334944]
        ],
        [
            [0.15239357635985891049884134685185, 0.17968185931866008747662175040473],
            [0.78694035922718786507195083272434, 0.27329165404555033518920991847707],
            [1.8933208179980234734221337551445, 0.17558597105229994960499431801123],
            [3.4865160580834006134354842606825, 0.055124701860867878869042603101124],
            [5.6302013071500465628581038196337, 0.0087372534350057720354131128807102],
            [8.4033444596058532780355056721105, 0.00069903230833599707663365344954055],
            [11.914845856823583540840891225631, 0.000026310654157878498770286149650435],
            [16.349170938816596871041071858974, 0.00000039616771027840191654270722273742],
            [22.065776998816441698055879604432, 0.0000000017164317799968611342076479311659],
            [29.989374900429517957858887566978, 0.00000000000092535226776880206925650102973008]
        ]
    ]

    _gf_weights_20 = [
        [0.075968817660977430171720338608218, 0.093580783579352771027591111243217],
        [0.39689586344285178223058182064917, 0.1791589107707744955479337149704],
        [0.96338559614556759124829770157145, 0.18942324180609036374458913519739],
        [1.7672851741343874245065061716631, 0.13457298596861634995120491877005],
        [2.811378360754400251569432788449, 0.066234172237674553154645081830422],
        [4.1078405749944870532815194690286, 0.023090720311438621660004332277493],
        [5.6704354580312393319073163916557, 0.005838335078447861941181718209],
        [7.511843531087685760682423329693, 0.001084447794378760660604086318311],
        [9.6457977987012577479597289850791, 0.00014785885411091834452803752452765],
        [12.089349892555650122866006232861, 0.000014638381919646374769751722109044],
        [14.864277739984744292937753029254, 0.0000010333504555720962210304058398219],
        [17.998482895422259709776466064882, 0.000000050717447651635746691318192403764],
        [21.528081763693307450433382154657, 0.0000000016733110910186913010305255682043],
        [25.500770968520504080877774785735, 0.000000000035469439226133057176179097278067],
        [29.981445109248836821088727545494, 0.00000000000045398972247297970353013479293197],
        [35.062181201220739775015724569631, 0.0000000000000032122945191934557275240758014242],
        [40.881685476350758082137464351086, 0.000000000000000011001725170464245146025551301751],
        [47.66852218954259646594354505947, 0.000000000000000000014665773261340314311671210576367],
        [55.859236367272462852917009077357, 0.0000000000000000000000050364218571581813683430709803827],
        [66.572776033103327331933948621436, 0.00000000000000000000000000015782472920067090821721193558893]
    ]

    #NB this will only work with Jurai's rates! So we inherit some params...
    def __init__(self, n_points, mp, H, tot=True):
        if not (1 <= n_points <= 10):
            raise Exception("Must have 1 <= n_points <= 10")

        _quad = np.array(self._gf_weights[n_points - 1])
        # _quad = np.array(self._gf_weights_20)
        self._kc_list = _quad[:, 0]
        # self._weights = _quad[:, 1]
        self._weights = _quad[:, 1] / f_nu(self._kc_list)
        self._rates = []

        # DEBUG
        print("kc_list: {}".format(self._kc_list))
        print("weights: {}".format(self._weights))

        rates_interface = Rates_Jurai(mp, H, self._kc_list, tot)

        for kc in self._kc_list:
            self._rates.append(rates_interface.get_rates(kc))

    def kc_list(self):
        return self._kc_list

    def weights(self):
        return self._weights

    def rates(self):
        return self._rates