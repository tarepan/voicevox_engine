"""/sing_frame_audio_query API のテスト。"""

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.utility import round_floats


def test_post_sing_frame_volume_200(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    score = {
        "notes": [
            {"key": None, "frame_length": 15, "lyric": ""},
            {"key": 60, "frame_length": 45, "lyric": "ド"},
            {"key": 62, "frame_length": 45, "lyric": "レ"},
            {"key": 64, "frame_length": 45, "lyric": "ミ"},
            {"key": None, "frame_length": 15, "lyric": ""},
        ]
    }

    frame_audio_query = {
        "f0": [
            203.12020874023438,
            203.22958374023438,
            203.18966674804688,
            203.2109832763672,
            203.19461059570312,
            203.1883087158203,
            203.20797729492188,
            203.20526123046875,
            203.19091796875,
            203.1884002685547,
            203.28414916992188,
            203.3013153076172,
            202.45860290527344,
            183.76019287109375,
            204.18850708007812,
            226.11729431152344,
            234.71311950683594,
            237.92971801757812,
            235.15191650390625,
            231.90707397460938,
            229.99266052246094,
            229.66290283203125,
            230.2884063720703,
            231.7293243408203,
            234.10093688964844,
            237.48324584960938,
            241.1653289794922,
            244.013916015625,
            246.09280395507812,
            248.35073852539062,
            251.46533203125,
            255.3215789794922,
            259.22845458984375,
            262.0407409667969,
            263.4653015136719,
            264.0202941894531,
            264.0643615722656,
            264.0140075683594,
            264.18475341796875,
            264.071533203125,
            263.0423889160156,
            261.5002746582031,
            260.30755615234375,
            259.74737548828125,
            259.97265625,
            260.8736877441406,
            261.92633056640625,
            262.47418212890625,
            262.64044189453125,
            262.7281188964844,
            262.550537109375,
            262.14971923828125,
            262.1888732910156,
            262.9635009765625,
            264.0044250488281,
            265.0442810058594,
            265.5814514160156,
            266.043701171875,
            270.6605224609375,
            278.0888366699219,
            286.00537109375,
            292.86920166015625,
            297.3049621582031,
            299.27734375,
            300.3929443359375,
            300.6061706542969,
            299.57958984375,
            297.44024658203125,
            294.6528015136719,
            291.82952880859375,
            289.404052734375,
            287.65704345703125,
            286.74700927734375,
            286.7215881347656,
            287.2806091308594,
            288.5699462890625,
            290.4000244140625,
            292.43536376953125,
            294.1000061035156,
            295.3788146972656,
            296.26678466796875,
            296.9929504394531,
            297.8184509277344,
            298.62176513671875,
            299.185302734375,
            299.1925964355469,
            298.5477294921875,
            297.5537109375,
            296.7957458496094,
            296.4950256347656,
            296.5893249511719,
            297.08233642578125,
            297.6444091796875,
            298.2401123046875,
            298.5912780761719,
            298.34576416015625,
            297.736083984375,
            297.81304931640625,
            297.0906982421875,
            291.0978088378906,
            281.9891357421875,
            282.7239990234375,
            285.04388427734375,
            285.44024658203125,
            289.8936767578125,
            303.32220458984375,
            312.589599609375,
            314.6080322265625,
            317.6407470703125,
            321.5991516113281,
            325.74896240234375,
            327.905517578125,
            328.0322265625,
            326.0955505371094,
            322.309326171875,
            319.3800354003906,
            319.07257080078125,
            319.984130859375,
            321.45501708984375,
            323.28179931640625,
            325.30828857421875,
            327.5245361328125,
            330.62957763671875,
            334.59619140625,
            338.2353515625,
            340.77130126953125,
            341.728759765625,
            340.83551025390625,
            338.66802978515625,
            335.49774169921875,
            330.9483642578125,
            325.6497497558594,
            321.55267333984375,
            319.8941345214844,
            320.3050231933594,
            322.0650634765625,
            324.57373046875,
            327.59454345703125,
            330.9676208496094,
            334.1003723144531,
            336.59857177734375,
            338.0997314453125,
            338.8575134277344,
            338.69305419921875,
            336.66839599609375,
            330.25885009765625,
            319.44140625,
            311.1017150878906,
            316.9340515136719,
            319.87322998046875,
            305.58660888671875,
            298.75933837890625,
            309.8575134277344,
            314.0906066894531,
            316.8128662109375,
            333.0500793457031,
            355.0403747558594,
            355.17584228515625,
            355.1917724609375,
            355.20037841796875,
            355.1678771972656,
            355.1257019042969,
            355.2730712890625,
            355.0054931640625,
            354.0846252441406,
        ],
        "volume": [
            0.0004787147045135498,
            0.0010405704379081726,
            0.0014588013291358948,
            0.0012274235486984253,
            0.0010905563831329346,
            0.0009400248527526855,
            0.0009405165910720825,
            0.0006652474403381348,
            0.0002709329128265381,
            -0.0002336353063583374,
            -0.0002986416220664978,
            0.0005924627184867859,
            0.003974780440330505,
            0.017938457429409027,
            0.039812393486499786,
            0.0595753975212574,
            0.08433514088392258,
            0.09954013675451279,
            0.10054653882980347,
            0.0972367525100708,
            0.09018714725971222,
            0.0851733461022377,
            0.07580415159463882,
            0.06871049106121063,
            0.06137121468782425,
            0.05629901587963104,
            0.05338696017861366,
            0.05235457420349121,
            0.052907101809978485,
            0.05187157168984413,
            0.05293390154838562,
            0.05417793244123459,
            0.05544605106115341,
            0.05646364018321037,
            0.05738266557455063,
            0.05892559885978699,
            0.05947083234786987,
            0.05872538313269615,
            0.0584573820233345,
            0.05720439925789833,
            0.05803307890892029,
            0.05679874122142792,
            0.05562245100736618,
            0.05525610223412514,
            0.05318000912666321,
            0.05259035527706146,
            0.05080951377749443,
            0.05003567039966583,
            0.0496600866317749,
            0.04906811937689781,
            0.04923130199313164,
            0.04877219721674919,
            0.0484158918261528,
            0.048542704433202744,
            0.04962582886219025,
            0.045066948980093,
            0.037152379751205444,
            0.03356718644499779,
            0.036367204040288925,
            0.05020733177661896,
            0.06368612498044968,
            0.07467876374721527,
            0.07978153228759766,
            0.08049686998128891,
            0.0795092061161995,
            0.07859177887439728,
            0.07510609179735184,
            0.07104776799678802,
            0.06727584451436996,
            0.06365916877985,
            0.06015634536743164,
            0.058127306401729584,
            0.05694492161273956,
            0.05727839097380638,
            0.05793873220682144,
            0.06043902784585953,
            0.06278379261493683,
            0.06741152703762054,
            0.06920691579580307,
            0.07085863500833511,
            0.0709768608212471,
            0.07126042246818542,
            0.07135792076587677,
            0.07060225307941437,
            0.0693637952208519,
            0.06815081089735031,
            0.06793922185897827,
            0.06794025748968124,
            0.06763045489788055,
            0.06797683238983154,
            0.0699392557144165,
            0.07168693840503693,
            0.07338269054889679,
            0.07539191842079163,
            0.07654494792222977,
            0.07708939164876938,
            0.07883097976446152,
            0.08011248707771301,
            0.0797126516699791,
            0.07450003921985626,
            0.06884774565696716,
            0.0707208514213562,
            0.08004502952098846,
            0.09337422251701355,
            0.1131061464548111,
            0.13021202385425568,
            0.14446872472763062,
            0.14968127012252808,
            0.15179219841957092,
            0.14891058206558228,
            0.1466791033744812,
            0.14219200611114502,
            0.1366046965122223,
            0.12968313694000244,
            0.12211640924215317,
            0.11668159067630768,
            0.11206194758415222,
            0.10870714485645294,
            0.10816323757171631,
            0.11024853587150574,
            0.11374224722385406,
            0.12013312429189682,
            0.12631694972515106,
            0.1320280134677887,
            0.13775652647018433,
            0.14057473838329315,
            0.14202912151813507,
            0.13998106122016907,
            0.1357124000787735,
            0.12879544496536255,
            0.11972789466381073,
            0.10934856534004211,
            0.10000964999198914,
            0.09129352122545242,
            0.08448395133018494,
            0.07895656675100327,
            0.07746186852455139,
            0.07585690915584564,
            0.07549901306629181,
            0.0747423842549324,
            0.07375279814004898,
            0.0723096951842308,
            0.06903150677680969,
            0.06309784948825836,
            0.05319780111312866,
            0.043854743242263794,
            0.03212194889783859,
            0.020242862403392792,
            0.010037116706371307,
            0.003522723913192749,
            0.001071922481060028,
            0.00034518539905548096,
            0.0006061941385269165,
            0.0007827728986740112,
            0.0006604194641113281,
            0.0008484125137329102,
            0.0013152658939361572,
            0.00047776103019714355,
            0.00029630959033966064,
            0.0005781650543212891,
            0.00028308480978012085,
            0.00037220120429992676,
            -8.973479270935059e-05,
            0.0009073689579963684,
            -0.001959472894668579,
        ],
        "phonemes": [
            {"phoneme": "pau", "frame_length": 13},
            {"phoneme": "d", "frame_length": 2},
            {"phoneme": "o", "frame_length": 41},
            {"phoneme": "r", "frame_length": 4},
            {"phoneme": "e", "frame_length": 40},
            {"phoneme": "m", "frame_length": 5},
            {"phoneme": "i", "frame_length": 45},
            {"phoneme": "pau", "frame_length": 15},
        ],
        "volumeScale": 1.0,
        "outputSamplingRate": 24000,
        "outputStereo": False,
    }

    response = client.post(
        "/sing_frame_volume",
        params={"speaker": 0},
        json={"score": score, "frame_audio_query": frame_audio_query},
    )
    print(response.text)
    assert response.status_code == 200
    assert snapshot_json == round_floats(response.json(), 2)
