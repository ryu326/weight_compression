#!/bin/bash
# for model in "mistralai/Mistral-7B-Instruct-v0.2"
# do
#     echo $model;
#     WANDB_MODE=disabled uv run python scripts/train_lora_baselines.py $model boolq winogrande piqa hellaswag arc_easy arc_challenge openbookqa gsm8k mbpp lol_022 lol_033 lol_034 lol_035 lol_039 lol_043 lol_044 lol_045 lol_047 lol_050 lol_063 lol_064 lol_065 lol_066 lol_067 lol_068 lol_069 lol_070 lol_072 lol_074 lol_075 lol_076 lol_077 lol_079 lol_080 lol_081 lol_083 lol_084 lol_085 lol_087 lol_089 lol_092 lol_093 lol_094 lol_095 lol_101 lol_102 lol_105 lol_107 lol_108 lol_1087 lol_1088 lol_1089 lol_110 lol_113 lol_1135 lol_1146 lol_1147 lol_1148 lol_1151 lol_1152 lol_1154 lol_1156 lol_1157 lol_1158 lol_116 lol_1167 lol_1168 lol_118 lol_1186 lol_1188 lol_1189 lol_119 lol_1190 lol_1192 lol_1194 lol_1196 lol_1197 lol_1198 lol_1199 lol_1200 lol_1201 lol_1203 lol_1204 lol_1206 lol_1207 lol_1209 lol_121 lol_1210 lol_1211 lol_1212 lol_1214 lol_1216 lol_1217 lol_123 lol_127 lol_128 lol_1283 lol_1285 lol_1286 lol_1288 lol_129 lol_1292 lol_1294 lol_130 lol_1308 lol_131 lol_1310 lol_1311 lol_1313 lol_1315 lol_1316 lol_1317 lol_1318 lol_1319 lol_1320 lol_1321 lol_1322 lol_1325 lol_1326 lol_1328 lol_1332 lol_1338 lol_1341 lol_1342 lol_1347 lol_1355 lol_137 lol_1378 lol_1379 lol_138 lol_1380 lol_1384 lol_1385 lol_1386 lol_1387 lol_1389 lol_1390 lol_1391 lol_1393 lol_1394 lol_1398 lol_140 lol_1400 lol_1401 lol_1403 lol_1404 lol_1406 lol_1409 lol_1418 lol_1419 lol_1420 lol_1421 lol_1425 lol_1427 lol_1428 lol_1429 lol_1431 lol_1434 lol_144 lol_1444 lol_1447 lol_1448 lol_1449 lol_145 lol_1451 lol_1452 lol_1453 lol_146 lol_147 lol_1479 lol_148 lol_1482 lol_1483 lol_1486 lol_1487 lol_1489 lol_149 lol_1495 lol_1502 lol_1503 lol_1504 lol_1506 lol_1508 lol_1509 lol_151 lol_1510 lol_1518 lol_1520 lol_1529 lol_153 lol_1533 lol_1534 lol_1541 lol_155 lol_1551 lol_1557 lol_1562 lol_1564 lol_1565 lol_1566 lol_1567 lol_1568 lol_157 lol_1572 lol_1581 lol_1582 lol_1583 lol_1584 lol_1585 lol_1590 lol_1592 lol_1593 lol_1596 lol_1598 lol_1599 lol_1601 lol_1603 lol_1605 lol_1606 lol_1607 lol_1609 lol_161 lol_162 lol_1622 lol_163 lol_1631 lol_1645 lol_1656 lol_1657 lol_1665 lol_1669 lol_167 lol_1670 lol_1703 lol_1704 lol_1711 lol_1712 lol_1713 lol_1714 lol_1720 lol_1721 lol_1722 lol_1723 lol_1724 lol_1728 lol_1729 lol_1731 lol_176 lol_178 lol_181 lol_183 lol_190 lol_192 lol_195 lol_201 lol_202 lol_206 lol_207 lol_209 lol_210 lol_211 lol_219 lol_228 lol_243 lol_244 lol_245 lol_246 lol_247 lol_249 lol_267 lol_269 lol_270 lol_274 lol_275 lol_277 lol_278 lol_280 lol_284 lol_285 lol_288 lol_290 lol_291 lol_294 lol_296 lol_298 lol_300 lol_304 lol_308 lol_316 lol_318 lol_319 lol_322 lol_323 lol_324 lol_325 lol_326 lol_328 lol_329 lol_330 lol_333 lol_335 lol_341 lol_343 lol_344 lol_346 lol_351 lol_353 lol_355 lol_356 lol_357 lol_359 lol_362 lol_363 lol_365 lol_366 lol_370 lol_377 lol_378 lol_379 lol_380 lol_381 lol_382 lol_385 lol_388 lol_389 lol_390 lol_391 lol_393 lol_398 lol_400 lol_403 lol_413 lol_428 lol_429 lol_431 lol_442 lol_453 lol_454 lol_455 lol_456 lol_457 lol_460 lol_461 lol_472 lol_475 lol_477 lol_488 lol_489 lol_492 lol_494 lol_495 lol_497 lol_499 lol_504 lol_505 lol_507 lol_509 lol_513 lol_515 lol_516 lol_517 lol_518 lol_550 lol_563 lol_564 lol_565 lol_566 lol_568 lol_574 lol_577 lol_579 lol_580 lol_582 lol_583 lol_584 lol_585 lol_587 lol_588 lol_590 lol_593 lol_594 lol_596 lol_600 lol_605 lol_607 lol_609 lol_614 lol_615 lol_616 lol_617 lol_618 lol_619 lol_620 lol_625 lol_627 lol_628 lol_629 lol_630 lol_632 lol_633 lol_636 lol_637 lol_638 lol_640 lol_641 lol_642 lol_645 lol_648 lol_664 lol_666 lol_667 lol_670 lol_671 lol_672 lol_674 lol_675 lol_679 lol_683 lol_685 lol_686 lol_687 lol_689 lol_691 lol_692 lol_694 lol_695 lol_696 lol_697 lol_698 lol_699 lol_700 lol_701 lol_703 lol_704 lol_705 lol_706 lol_707 lol_708 lol_710 lol_713 lol_714 lol_716 lol_717 lol_719 lol_720 lol_721 lol_722 lol_723 lol_724 lol_725 lol_726 lol_727 lol_728 lol_732 lol_733 lol_734 lol_736 lol_740 lol_742 lol_746 lol_750 lol_751 lol_753 lol_754 lol_755 lol_761 lol_769 lol_770 lol_819 lol_821 lol_828 lol_833 lol_834 lol_846 lol_850 lol_852 lol_856 lol_858 lol_859 lol_861 lol_865 lol_867 lol_874 lol_875 lol_879 lol_889 lol_890 lol_891 lol_892 lol_893 lol_901 lol_903 lol_904 lol_905 lol_908 lol_923 lol_924 lol_925 lol_926 lol_927 lol_929 lol_933 lol_934 lol_936 lol_955 lol_956 lol_963 lol_964 lol_966
# done

for model in "mistralai/Mistral-7B-Instruct-v0.2"
do
    echo $model;
    CUDA_VISIBLE_DEVICES=3 WANDB_MODE=disabled uv run python scripts/train_lora_baselines.py $model piqa hellaswag
done


# TASKS=(boolq winogrande piqa hellaswag arc_easy arc_challenge openbookqa gsm8k mbpp lol_022 lol_033 lol_034 lol_035 lol_039 lol_043 lol_044 lol_045 lol_047 lol_050 lol_063 lol_064 lol_065 lol_066 lol_067 lol_068 lol_069 lol_070 lol_072 lol_074 lol_075 lol_076 lol_077 lol_079 lol_080 lol_081 lol_083 lol_084 lol_085 lol_087 lol_089 lol_092 lol_093 lol_094 lol_095 lol_101 lol_102 lol_105 lol_107 lol_108 lol_1087 lol_1088 lol_1089 lol_110 lol_113 lol_1135 lol_1146 lol_1147 lol_1148 lol_1151 lol_1152 lol_1154 lol_1156 lol_1157 lol_1158 lol_116 lol_1167 lol_1168 lol_118 lol_1186 lol_1188 lol_1189 lol_119 lol_1190 lol_1192 lol_1194 lol_1196 lol_1197 lol_1198 lol_1199 lol_1200 lol_1201 lol_1203 lol_1204 lol_1206 lol_1207 lol_1209 lol_121 lol_1210 lol_1211 lol_1212 lol_1214 lol_1216 lol_1217 lol_123 lol_127 lol_128 lol_1283 lol_1285 lol_1286 lol_1288 lol_129 lol_1292 lol_1294 lol_130 lol_1308 lol_131 lol_1310 lol_1311 lol_1313 lol_1315 lol_1316 lol_1317 lol_1318 lol_1319 lol_1320 lol_1321 lol_1322 lol_1325 lol_1326 lol_1328 lol_1332 lol_1338 lol_1341 lol_1342 lol_1347 lol_1355 lol_137 lol_1378 lol_1379 lol_138 lol_1380 lol_1384 lol_1385 lol_1386 lol_1387 lol_1389 lol_1390 lol_1391 lol_1393 lol_1394 lol_1398 lol_140 lol_1400 lol_1401 lol_1403 lol_1404 lol_1406 lol_1409 lol_1418 lol_1419 lol_1420 lol_1421 lol_1425 lol_1427 lol_1428 lol_1429 lol_1431 lol_1434 lol_144 lol_1444 lol_1447 lol_1448 lol_1449 lol_145 lol_1451 lol_1452 lol_1453 lol_146 lol_147 lol_1479 lol_148 lol_1482 lol_1483 lol_1486 lol_1487 lol_1489 lol_149 lol_1495 lol_1502 lol_1503 lol_1504 lol_1506 lol_1508 lol_1509 lol_151 lol_1510 lol_1518 lol_1520 lol_1529 lol_153 lol_1533 lol_1534 lol_1541 lol_155 lol_1551 lol_1557 lol_1562 lol_1564 lol_1565 lol_1566 lol_1567 lol_1568 lol_157 lol_1572 lol_1581 lol_1582 lol_1583 lol_1584 lol_1585 lol_1590 lol_1592 lol_1593 lol_1596 lol_1598 lol_1599 lol_1601 lol_1603 lol_1605 lol_1606 lol_1607 lol_1609 lol_161 lol_162 lol_1622 lol_163 lol_1631 lol_1645 lol_1656 lol_1657 lol_1665 lol_1669 lol_167 lol_1670 lol_1703 lol_1704 lol_1711 lol_1712 lol_1713 lol_1714 lol_1720 lol_1721 lol_1722 lol_1723 lol_1724 lol_1728 lol_1729 lol_1731 lol_176 lol_178 lol_181 lol_183 lol_190 lol_192 lol_195 lol_201 lol_202 lol_206 lol_207 lol_209 lol_210 lol_211 lol_219 lol_228 lol_243 lol_244 lol_245 lol_246 lol_247 lol_249 lol_267 lol_269 lol_270 lol_274 lol_275 lol_277 lol_278 lol_280 lol_284 lol_285 lol_288 lol_290 lol_291 lol_294 lol_296 lol_298 lol_300 lol_304 lol_308 lol_316 lol_318 lol_319 lol_322 lol_323 lol_324 lol_325 lol_326 lol_328 lol_329 lol_330 lol_333 lol_335 lol_341 lol_343 lol_344 lol_346 lol_351 lol_353 lol_355 lol_356 lol_357 lol_359 lol_362 lol_363 lol_365 lol_366 lol_370 lol_377 lol_378 lol_379 lol_380 lol_381 lol_382 lol_385 lol_388 lol_389 lol_390 lol_391 lol_393 lol_398 lol_400 lol_403 lol_413 lol_428 lol_429 lol_431 lol_442 lol_453 lol_454 lol_455 lol_456 lol_457 lol_460 lol_461 lol_472 lol_475 lol_477 lol_488 lol_489 lol_492 lol_494 lol_495 lol_497 lol_499 lol_504 lol_505 lol_507 lol_509 lol_513 lol_515 lol_516 lol_517 lol_518 lol_550 lol_563 lol_564 lol_565 lol_566 lol_568 lol_574 lol_577 lol_579 lol_580 lol_582 lol_583 lol_584 lol_585 lol_587 lol_588 lol_590 lol_593 lol_594 lol_596 lol_600 lol_605 lol_607 lol_609 lol_614 lol_615 lol_616 lol_617 lol_618 lol_619 lol_620 lol_625 lol_627 lol_628 lol_629 lol_630 lol_632 lol_633 lol_636 lol_637 lol_638 lol_640 lol_641 lol_642 lol_645 lol_648 lol_664 lol_666 lol_667 lol_670 lol_671 lol_672 lol_674 lol_675 lol_679 lol_683 lol_685 lol_686 lol_687 lol_689 lol_691 lol_692 lol_694 lol_695 lol_696 lol_697 lol_698 lol_699 lol_700 lol_701 lol_703 lol_704 lol_705 lol_706 lol_707 lol_708 lol_710 lol_713 lol_714 lol_716 lol_717 lol_719 lol_720 lol_721 lol_722 lol_723 lol_724 lol_725 lol_726 lol_727 lol_728 lol_732 lol_733 lol_734 lol_736 lol_740 lol_742 lol_746 lol_750 lol_751 lol_753 lol_754 lol_755 lol_761 lol_769 lol_770 lol_819 lol_821 lol_828 lol_833 lol_834 lol_846 lol_850 lol_852 lol_856 lol_858 lol_859 lol_861 lol_865 lol_867 lol_874 lol_875 lol_879 lol_889 lol_890 lol_891 lol_892 lol_893 lol_901 lol_903 lol_904 lol_905 lol_908 lol_923 lol_924 lol_925 lol_926 lol_927 lol_929 lol_933 lol_934 lol_936 lol_955 lol_956 lol_963 lol_964 lol_966)
# TASKS=(winogrande piqa hellaswag arc_easy arc_challenge openbookqa gsm8k mbpp lol_022 lol_033 lol_034 lol_035 lol_039 lol_043 lol_044 lol_045 lol_047 lol_050 lol_063 lol_064 lol_065 lol_066 lol_067 lol_068 lol_069 lol_070 lol_072 lol_074 lol_075 lol_076 lol_077 lol_079 lol_080 lol_081 lol_083 lol_084 lol_085 lol_087 lol_089 lol_092 lol_093 lol_094 lol_095 lol_101 lol_102 lol_105 lol_107 lol_108 lol_1087 lol_1088 lol_1089 lol_110 lol_113 lol_1135 lol_1146 lol_1147 lol_1148 lol_1151 lol_1152 lol_1154 lol_1156 lol_1157 lol_1158 lol_116 lol_1167 lol_1168 lol_118 lol_1186 lol_1188 lol_1189 lol_119 lol_1190 lol_1192 lol_1194 lol_1196 lol_1197 lol_1198 lol_1199 lol_1200 lol_1201 lol_1203 lol_1204 lol_1206 lol_1207 lol_1209 lol_121 lol_1210 lol_1211 lol_1212 lol_1214 lol_1216 lol_1217 lol_123 lol_127 lol_128 lol_1283 lol_1285 lol_1286 lol_1288 lol_129 lol_1292 lol_1294 lol_130 lol_1308 lol_131 lol_1310 lol_1311 lol_1313 lol_1315 lol_1316 lol_1317 lol_1318 lol_1319 lol_1320 lol_1321 lol_1322 lol_1325 lol_1326 lol_1328 lol_1332 lol_1338 lol_1341 lol_1342 lol_1347 lol_1355 lol_137 lol_1378 lol_1379 lol_138 lol_1380 lol_1384 lol_1385 lol_1386 lol_1387 lol_1389 lol_1390 lol_1391 lol_1393 lol_1394 lol_1398 lol_140 lol_1400 lol_1401 lol_1403 lol_1404 lol_1406 lol_1409 lol_1418 lol_1419 lol_1420 lol_1421 lol_1425 lol_1427 lol_1428 lol_1429 lol_1431 lol_1434 lol_144 lol_1444 lol_1447 lol_1448 lol_1449 lol_145 lol_1451 lol_1452 lol_1453 lol_146 lol_147 lol_1479 lol_148 lol_1482 lol_1483 lol_1486 lol_1487 lol_1489 lol_149 lol_1495 lol_1502 lol_1503 lol_1504 lol_1506 lol_1508 lol_1509 lol_151 lol_1510 lol_1518 lol_1520 lol_1529 lol_153 lol_1533 lol_1534 lol_1541 lol_155 lol_1551 lol_1557 lol_1562 lol_1564 lol_1565 lol_1566 lol_1567 lol_1568 lol_157 lol_1572 lol_1581 lol_1582 lol_1583 lol_1584 lol_1585 lol_1590 lol_1592 lol_1593 lol_1596 lol_1598 lol_1599 lol_1601 lol_1603 lol_1605 lol_1606 lol_1607 lol_1609 lol_161 lol_162 lol_1622 lol_163 lol_1631 lol_1645 lol_1656 lol_1657 lol_1665 lol_1669 lol_167 lol_1670 lol_1703 lol_1704 lol_1711 lol_1712 lol_1713 lol_1714 lol_1720 lol_1721 lol_1722 lol_1723 lol_1724 lol_1728 lol_1729 lol_1731 lol_176 lol_178 lol_181 lol_183 lol_190 lol_192 lol_195 lol_201 lol_202 lol_206 lol_207 lol_209 lol_210 lol_211 lol_219 lol_228 lol_243 lol_244 lol_245 lol_246 lol_247 lol_249 lol_267 lol_269 lol_270 lol_274 lol_275 lol_277 lol_278 lol_280 lol_284 lol_285 lol_288 lol_290 lol_291 lol_294 lol_296 lol_298 lol_300 lol_304 lol_308 lol_316 lol_318 lol_319 lol_322 lol_323 lol_324 lol_325 lol_326 lol_328 lol_329 lol_330 lol_333 lol_335 lol_341 lol_343 lol_344 lol_346 lol_351 lol_353 lol_355 lol_356 lol_357 lol_359 lol_362 lol_363 lol_365 lol_366 lol_370 lol_377 lol_378 lol_379 lol_380 lol_381 lol_382 lol_385 lol_388 lol_389 lol_390 lol_391 lol_393 lol_398 lol_400 lol_403 lol_413 lol_428 lol_429 lol_431 lol_442 lol_453 lol_454 lol_455 lol_456 lol_457 lol_460 lol_461 lol_472 lol_475 lol_477 lol_488 lol_489 lol_492 lol_494 lol_495 lol_497 lol_499 lol_504 lol_505 lol_507 lol_509 lol_513 lol_515 lol_516 lol_517 lol_518 lol_550 lol_563 lol_564 lol_565 lol_566 lol_568 lol_574 lol_577 lol_579 lol_580 lol_582 lol_583 lol_584 lol_585 lol_587 lol_588 lol_590 lol_593 lol_594 lol_596 lol_600 lol_605 lol_607 lol_609 lol_614 lol_615 lol_616 lol_617 lol_618 lol_619 lol_620 lol_625 lol_627 lol_628 lol_629 lol_630 lol_632 lol_633 lol_636 lol_637 lol_638 lol_640 lol_641 lol_642 lol_645 lol_648 lol_664 lol_666 lol_667 lol_670 lol_671 lol_672 lol_674 lol_675 lol_679 lol_683 lol_685 lol_686 lol_687 lol_689 lol_691 lol_692 lol_694 lol_695 lol_696 lol_697 lol_698 lol_699 lol_700 lol_701 lol_703 lol_704 lol_705 lol_706 lol_707 lol_708 lol_710 lol_713 lol_714 lol_716 lol_717 lol_719 lol_720 lol_721 lol_722 lol_723 lol_724 lol_725 lol_726 lol_727 lol_728 lol_732 lol_733 lol_734 lol_736 lol_740 lol_742 lol_746 lol_750 lol_751 lol_753 lol_754 lol_755 lol_761 lol_769 lol_770 lol_819 lol_821 lol_828 lol_833 lol_834 lol_846 lol_850 lol_852 lol_856 lol_858 lol_859 lol_861 lol_865 lol_867 lol_874 lol_875 lol_879 lol_889 lol_890 lol_891 lol_892 lol_893 lol_901 lol_903 lol_904 lol_905 lol_908 lol_923 lol_924 lol_925 lol_926 lol_927 lol_929 lol_933 lol_934 lol_936 lol_955 lol_956 lol_963 lol_964 lol_966)

# set -euo pipefail

# # ===== 설정 =====
# LOGDIR="logs"
# mkdir -p "$LOGDIR"

# # 실행할 모델들
# MODELS=("mistralai/Mistral-7B-Instruct-v0.2")

# # TASKS=(lol_587 lol_564 lol_1341 lol_201 lol_492 lol_499 lol_1342 lol_1294 lol_574 lol_130 lol_515 lol_582 lol_455 lol_1332 lol_1292 lol_137 lol_1206 lol_457 lol_1308 lol_1310 lol_1318 lol_123 lol_1316 lol_1378 lol_475 lol_1315 lol_192 lol_563 lol_1212 lol_495 lol_1285 lol_585 lol_477 lol_566 lol_1210 lol_1203 lol_1355 lol_1209 lol_505 lol_1321 lol_454 lol_504 lol_431 lol_1317 lol_131 lol_1325 lol_1216 lol_1347 lol_1328 lol_460 lol_489 lol_590 lol_577 lol_497 lol_488 lol_1288 lol_128 lol_580 lol_1311 lol_516 lol_593 lol_1326 lol_584 lol_596 lol_568 lol_1201 lol_1204 lol_565 lol_579 lol_494 lol_461 lol_1313 lol_1211 lol_518 lol_513 lol_121 lol_195 lol_1338 lol_1320 lol_1283 lol_588 lol_129 lol_1200 lol_583 lol_127 lol_1214 lol_1319 lol_456 lol_517 lol_472 lol_507 lol_550 lol_509 lol_1322 lol_594)
# TASKS=(piqa hellaswag)
# # ===== 함수: 태스크 4등분 후 GPU 0~3에 배치 =====
# run_parallel_for_model () {
#   local model="$1"
#   shift

#   # 모델명을 안전한 파일명으로
#   local safe_model
#   safe_model="$(echo -n "$model" | tr '/:' '__')"

#   local -a tasks=("$@")
#   local total=${#tasks[@]}
#   local groups=2
#   local per_group=$(( (total + groups - 1) / groups ))  # 올림 나누기

#   echo "[INFO] model=$model, total_tasks=$total, per_group=$per_group"

#   # 4개 작업 병렬 실행
#   for gpu in 2 3; do
#     local start=$(( gpu * per_group ))
#     if (( start >= total )); then
#       echo "[INFO] GPU $gpu: 할당할 태스크 없음"
#       continue
#     fi
#     local chunk=("${tasks[@]:start:per_group}")

#     # 로그 파일
#     local logfile="${LOGDIR}/${safe_model}_gpu${gpu}.log"

#     echo "[LAUNCH] GPU ${gpu}: ${#chunk[@]} tasks -> ${logfile}"

#     CUDA_VISIBLE_DEVICES="${gpu}" WANDB_MODE=disabled \
#       stdbuf -oL -eL uv run python scripts/train_lora_baselines.py \
#       "$model" "${chunk[@]}" \
#       >"$logfile" 2>&1 &
#   done

#   # 모든 백그라운드 잡 종료 대기
# #   wait
# #   echo "[DONE] model=$model"
# }

# # ===== 메인 루프 =====
# for model in "${MODELS[@]}"; do
#   run_parallel_for_model "$model" "${TASKS[@]}"
# done

# set -euo pipefail

# # ===== 설정 =====
# LOGDIR="logs"
# mkdir -p "$LOGDIR"

# # 실행할 모델들
# MODELS=("mistralai/Mistral-7B-Instruct-v0.2")

# # 작업 목록 (앞에서부터 순서 유지)
# TASKS=(lol_587 lol_564 lol_1341 lol_201 lol_492 lol_499 lol_1342 lol_1294 lol_574 lol_130 lol_515 lol_582 lol_455 lol_1332 lol_1292 lol_137 lol_1206 lol_457 lol_1308 lol_1310 lol_1318 lol_123 lol_1316 lol_1378 lol_475 lol_1315 lol_192 lol_563 lol_1212 lol_495 lol_1285 lol_585 lol_477 lol_566 lol_1210 lol_1203 lol_1355 lol_1209 lol_505 lol_1321 lol_454 lol_504 lol_431 lol_1317 lol_131 lol_1325 lol_1216 lol_1347 lol_1328 lol_460 lol_489 lol_590 lol_577 lol_497 lol_488 lol_1288 lol_128 lol_580 lol_1311 lol_516 lol_593 lol_1326 lol_584 lol_596 lol_568 lol_1201 lol_1204 lol_565 lol_579 lol_494 lol_461 lol_1313 lol_1211 lol_518 lol_513 lol_121 lol_195 lol_1338 lol_1320 lol_1283 lol_588 lol_129 lol_1200 lol_583 lol_127 lol_1214 lol_1319 lol_456 lol_517 lol_472 lol_507 lol_550 lol_509 lol_1322 lol_594)

# # ===== 함수: FIFO 큐로 순서 보장 + GPU 0~2 병렬 처리 =====
# run_sequential_parallel () {
#   local model="$1"; shift
#   local -a tasks=("$@")

#   local safe_model
#   safe_model="$(echo -n "$model" | tr '/:' '__')"

#   echo "[INFO] model=$model, total_tasks=${#tasks[@]}"

#   # FIFO 생성
#   local fifo="/tmp/task_fifo_$$"
#   mkfifo "$fifo"

#   # 작업을 FIFO에 순서대로 밀어넣기
#   (
#     for t in "${tasks[@]}"; do
#       printf '%s\n' "$t"
#     done
#   ) > "$fifo" &

#   # GPU 워커 3개 실행 (각 워커는 FIFO에서 순서대로 하나씩 꺼내 처리)
#   for gpu in 0 1 2; do
#     {
#       while IFS= read -r task || [[ -n "${task:-}" ]]; do
#         [[ -z "$task" ]] && continue
#         logfile="${LOGDIR}/${safe_model}_gpu${gpu}_${task}.log"
#         echo "[LAUNCH] GPU ${gpu}: $task -> $logfile"
#         CUDA_VISIBLE_DEVICES="${gpu}" WANDB_MODE=disabled \
#           stdbuf -oL -eL uv run python scripts/train_lora_baselines.py \
#           "$model" "$task" \
#           >"$logfile" 2>&1
#       done < "$fifo"
#     } &
#   done

#   # 모든 워커 종료 대기
#   wait
#   rm -f "$fifo"
#   echo "[DONE] model=$model"
# }

# # ===== 메인 =====
# for model in "${MODELS[@]}"; do
#   run_sequential_parallel "$model" "${TASKS[@]}"
# done