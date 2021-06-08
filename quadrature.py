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

class GaussFermiDiracQuadrature(Quadrature):

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
        ],
        [
            [0.1385747378442939857866483054933, 0.16471245961857565040937621206705],
            [0.7170808382243067262364483982965, 0.26160576892504818801432764086847],
            [1.7267288535858845118944023284354, 0.18463089157227529291350798863983],
            [3.1734292335016687004156045173785, 0.067477627043959041516526724859224],
            [5.1048803563897555592032062381861, 0.01322609564367359042383399807825],
            [7.5806312469858900101440087108988, 0.0014127049596772641478929804478176],
            [10.674306720690985971910602940035, 0.000079466313441372018149048676371454],
            [14.498715521044113087590498767539, 0.0000021431460509888907700054280431133],
            [19.245192596869490019713303347599, 0.000000023265007753142936733194325024871],
            [25.283522297127211277850446936333, 0.000000000072209157329410594239860650711753],
            [33.562613670554337757142871526274, 0.000000000000027010610500194958944090329750734]
        ],
        [
            [0.12702627534240287720632308217661, 0.15200002388304401906817903469516],
            [0.65848659382167367085168511750495, 0.25018174875262449222011825144356],
            [1.5871986724137102692820911398467, 0.19078296569126836129489508826379],
            [2.9133113862997038077091245581337, 0.079061737870996573347786405469908],
            [4.6729785726052338189014937275322, 0.018441570837096769414455697293538],
            [6.9132044288297283213762410069183, 0.0024810171343740798826670590799916],
            [9.6873495911076222593326832069629, 0.00019011950172694803788518020269946],
            [13.070713835628547678353325725394, 0.0000078391178743091189779434950894224],
            [17.180694236416079291622178445654, 0.00000015650810647620617806670674455445],
            [22.214383110377113137572834036588, 0.0000000012599576907916695791815581010622],
            [28.55086639495060975507453280291, 0.000000000002874824683596238858534815191565],
            [37.161711828610827228974744844425, 0.000000000000000765350823576767604333041458213]
        ],
        [
            [0.11723410183100475743087432427471, 0.1410762111497641428302768430185],
            [0.60864414476623091488912814007503, 0.23923914929404519939958139034332],
            [1.4685594596383577829883658095405, 0.19465138977750820694346880560795],
            [2.6934762080002589183684780559695, 0.089649718462356235766139728898229],
            [4.3109285443630071167238503465437, 0.024188376121710951504541051669801],
            [6.3591725282974892402652089831456, 0.0039335515214928151120420498513621],
            [8.8790242860512880441007466587054, 0.00038600303427394483389176064189477],
            [11.923921465650292351620803516083, 0.000022082834516790741522508177563845],
            [15.572400772180793771400653359873, 0.00000068786990110908778111386651232326],
            [19.945698455578947118193181479774, 0.000000010430547018309369024870206499827],
            [25.245919427767657950726339327348, 0.000000000063719673059213159433438244878354],
            [31.860735774578623151633810705687, 0.00000000000010920068850810782944600413666095],
            [40.783109957496828165221053472216, 0.000000000000000021140896577249946098587052423304]
        ],
        [
            [0.10882795516914038155654661932193, 0.13159256056663563012771338466671],
            [0.56573593071768084474766747565789, 0.22888057381770776450493627783852],
            [1.3664054132544418588272182566505, 0.19673824994633312053068264076173],
            [2.5050516885983538237470205723728, 0.099138447448965733516550079421939],
            [4.0026336440720307114410330377925, 0.030275575438883184737663627292746],
            [5.8909261121532442228460168799689, 0.0057733679242566256133638737796781],
            [8.2025111567599708756683951861115, 0.00069442485632879384983855190954548],
            [10.97712765633432591785749899842, 0.000051655725593211102111798923436832],
            [14.27048917140992443279012022774, 0.0000022695082048310612236350650311997],
            [18.163967578175485275536522145337, 0.000000054680953697853775407493299114237],
            [22.782031946145565564032460521795, 0.00000000064304019854805566122337438695333],
            [28.331318091336043429906727941467, 0.0000000000030385344010473516748566286046668],
            [35.207535962759414881821750849653, 0.0000000000000039829990838124572250339535686692],
            [44.423970406112592942413343954217, 0.00000000000000000057118601895008779098642806663424]
        ],
        [
            [0.10153449474240816043843043661205, 0.12328479300519438141447653006282],
            [0.52841473341786258588393557022264, 0.21914196939606216686375237402356],
            [1.2775006089857401559307664920946, 0.19744738103215358131577789529686],
            [2.3416313601758204083522109239455, 0.10750849280662523518488044955737],
            [3.736677838673560594660598802191, 0.036531960340488609604252892352665],
            [5.4893843743877690084716907224957, 0.0079817836726388081672380003481982],
            [7.6266521776188687340257961208195, 0.0011392632658550430390114878151052],
            [10.17917884716605448603815903742, 0.00010522799284030154152723766796036],
            [13.188218699297450074104211742472, 0.0000060948113252725868466798072000887],
            [16.711699701194154024425869070567, 0.00000021020608244646425001156632706623],
            [20.833325927085160096210354077686, 0.000000003993479898278182027664791870639],
            [25.680416061894900250722446599305, 0.000000000037061747906221945600014910151863],
            [31.463792472265267179843424738737, 0.00000000000013767682329205230253464959432958],
            [38.586757987066703299767840861838, 0.00000000000000014021238706263002009378611601264],
            [48.081990520397562807998510968469, 0.000000000000000000015135474762759487006589635165218]
        ],
        [
            [0.09514760195581037004812227837573, 0.11594904079824637660045297428611],
            [0.49566067484460081882999074831859, 0.21002175606703468370545003567214],
            [1.1994113365875023019589090547833, 0.19709913798958028495211203272226],
            [2.198464642407804977265832605981, 0.11479352292466519396931911121021],
            [3.5047157413189185278340824177784, 0.042812938658134610598289589555543],
            [5.1408668801051576166199713626242, 0.010524340957757652528706648937431],
            [7.1297206458584953987341265971996, 0.0017390060036138112128245385853415],
            [9.4957673993832923417622910102996, 0.00019274631553807306363864847078363],
            [12.270587439183705024532349572071, 0.000014027027942176691792513435813565],
            [15.497209385158521296595565764099, 0.00000064574643493134143749428734431152],
            [19.23537257091837722734541811398, 0.000000017798044167290326414227757447095],
            [23.570783372077546110240482741781, 0.00000000027093345077145004388583340026389],
            [28.633351570669046398061377624976, 0.000000000002013927684770142223417192883787],
            [34.63781951229616309695217044464, 0.000000000000005964222600825110661343427352938],
            [41.994709045072290864425561422309, 0.0000000000000000047836668877396492988578163797598],
            [51.755273302994286640858173545023, 0.00000000000000000000039422110789052215772270016074401]
        ],
        [
            [0.09514760195581037004812227837573, 0.11594904079824637660045297428611],
            [0.49566067484460081882999074831859, 0.21002175606703468370545003567214],
            [1.1994113365875023019589090547833, 0.19709913798958028495211203272226],
            [2.198464642407804977265832605981, 0.11479352292466519396931911121021],
            [3.5047157413189185278340824177784, 0.042812938658134610598289589555543],
            [5.1408668801051576166199713626242, 0.010524340957757652528706648937431],
            [7.1297206458584953987341265971996, 0.0017390060036138112128245385853415],
            [9.4957673993832923417622910102996, 0.00019274631553807306363864847078363],
            [12.270587439183705024532349572071, 0.000014027027942176691792513435813565],
            [15.497209385158521296595565764099, 0.00000064574643493134143749428734431152],
            [19.23537257091837722734541811398, 0.000000017798044167290326414227757447095],
            [23.570783372077546110240482741781, 0.00000000027093345077145004388583340026389],
            [28.633351570669046398061377624976, 0.000000000002013927684770142223417192883787],
            [34.63781951229616309695217044464, 0.000000000000005964222600825110661343427352938],
            [41.994709045072290864425561422309, 0.0000000000000000047836668877396492988578163797598],
            [51.755273302994286640858173545023, 0.00000000000000000000039422110789052215772270016074401]
        ],
        [
            [0.084494960748159655437740753566968, 0.10358784703110381796581245948758],
            [0.44087919821904686634868962207217, 0.19353772496154365574149538326949],
            [1.0686180533343608188472785192767, 0.19418453004013936509480053978751],
            [1.9592993738259248527317751462046, 0.12638476321032378813800992329138],
            [3.1192197378980944156207079142314, 0.055009486491537956506387108858134],
            [4.5649580276949846358913646119136, 0.016428810168446321393677456513774],
            [6.313831188331397113332207146646, 0.0034455576492110754337957999388519],
            [8.3827292012657384149228857706548, 0.0005112513309478157375492876272755],
            [10.791643412378444480813519002006, 0.000053214689594863738651454143020313],
            [13.566544569763502435447157141144, 0.0000038076705346126405276253675147436],
            [16.7416822335842462390638465214, 0.00000018167389002914042788749338466819],
            [20.362777275433227501853767677846, 0.0000000055398705010671280246583414432392],
            [24.492417172304727310232611223372, 0.00000000010176201316906860876037189525959],
            [29.219732640119981189662328467219, 0.0000000000010343667877546225102396937932599],
            [34.679282293290185963940381027389, 0.0000000000000051168827008035931759033111132907],
            [41.093033662465900725814659666614, 0.0000000000000000099743088211644421285775391244448],
            [48.88502221324852554526030556703, 0.0000000000000000000051360592121515675761103559962797],
            [59.141539999028191487559138987993, 0.00000000000000000000000025578154557360598319690689912076]
        ],
        [
            [0.080007747757089654161763673232951, 0.098333809741993387414989281383998],
            [0.41774653528340564314660301348199, 0.1861041903310452254745544751463],
            [1.0132995397882781935340898703568, 0.19197019591721367688449484572172],
            [1.8583267697095772029440852826883, 0.13086019777181336351325986132307],
            [2.9571664095754669354221077792254, 0.060768952383295286322584134873689],
            [4.3240255103715638994734074082513, 0.019690083582231911622989724332612],
            [5.9742802573623972362671870902719, 0.0045580967703731836867498110233965],
            [7.9224328612392318801163051694021, 0.00076183625282518896394019694790425],
            [10.184902230133823071261302015806, 0.00009154814690600531808052207430893],
            [12.782549845460098884924494099475, 0.0000077921784956526654361201702618446],
            [15.742409579866898566863644658612, 0.00000045891316610752312296547370023807],
            [19.099743105020494235877333755285, 0.000000018105603552146143516412523213256],
            [22.901315988415114369920072528522, 0.00000000045793567991321941692431111304462],
            [27.210917148572532842113867793695, 0.0000000000069878541100235207281029795558854],
            [32.119197109166858479134115506865, 0.000000000000058991710248485230104665218046332],
            [37.762840161591792368645699654952, 0.00000000000000024175917435547747362369802870817],
            [44.367180030833248775543552220244, 0.00000000000000000038805839124346778990332951244104],
            [52.362618803678868489414709789325, 0.00000000000000000000016249058130910452004470673852371],
            [62.852047171992722840290110939137, 0.0000000000000000000000000063899225477407555817789497619697]
        ],
        [
            [0.080007747757089654161763673232951, 0.098333809741993387414989281383998],
            [0.41774653528340564314660301348199, 0.1861041903310452254745544751463],
            [1.0132995397882781935340898703568, 0.19197019591721367688449484572172],
            [1.8583267697095772029440852826883, 0.13086019777181336351325986132307],
            [2.9571664095754669354221077792254, 0.060768952383295286322584134873689],
            [4.3240255103715638994734074082513, 0.019690083582231911622989724332612],
            [5.9742802573623972362671870902719, 0.0045580967703731836867498110233965],
            [7.9224328612392318801163051694021, 0.00076183625282518896394019694790425],
            [10.184902230133823071261302015806, 0.00009154814690600531808052207430893],
            [12.782549845460098884924494099475, 0.0000077921784956526654361201702618446],
            [15.742409579866898566863644658612, 0.00000045891316610752312296547370023807],
            [19.099743105020494235877333755285, 0.000000018105603552146143516412523213256],
            [22.901315988415114369920072528522, 0.00000000045793567991321941692431111304462],
            [27.210917148572532842113867793695, 0.0000000000069878541100235207281029795558854],
            [32.119197109166858479134115506865, 0.000000000000058991710248485230104665218046332],
            [37.762840161591792368645699654952, 0.00000000000000024175917435547747362369802870817],
            [44.367180030833248775543552220244, 0.00000000000000000038805839124346778990332951244104],
            [52.362618803678868489414709789325, 0.00000000000000000000016249058130910452004470673852371],
            [62.852047171992722840290110939137, 0.0000000000000000000000000063899225477407555817789497619697]
        ]
    ]
    #NB this will only work with Jurai's rates! So we inherit some params...
    def __init__(self, n_points, mp, H, tot=True):

        _quad = np.array(self._gf_weights[n_points - 1])

        self._kc_list = _quad[:, 0]
        self._weights = _quad[:, 1] / f_nu(self._kc_list)
        self._rates = []

        rates_interface = Rates_Jurai(mp, H, self._kc_list, tot)

        for kc in self._kc_list:
            self._rates.append(rates_interface.get_rates(kc))

    def kc_list(self):
        return self._kc_list

    def weights(self):
        return self._weights

    def rates(self):
        return self._rates

class GaussLegendreQuadrature(Quadrature):

    def __init__(self, n_points, kc_min, kc_max, mp, H, tot=True):

        gq_points, gq_weights = np.polynomial.legendre.leggauss(n_points)

        self._weights = gq_weights

        self._kc_list = np.array(list(map(
            lambda x: 0.5*(kc_max - kc_min)*x + 0.5*(kc_max + kc_min),
            gq_points
        )))

        rates_interface = Rates_Jurai(mp, H, self._kc_list, tot)

        self._rates = []
        for kc in self._kc_list:
            self._rates.append(rates_interface.get_rates(kc))

    def kc_list(self):
        return self._kc_list

    def weights(self):
        return self._weights

    def rates(self):
        return self._rates