#!/bin/bash
#SBATCH --job-name=aml3_s_eval_models
#SBATCH --qos=long
#SBATCH --time=0-15:00
#SBATCH --nodes=1
#SBATCH --partition=amd-rome
#SBATCH --output=/vast/home/jomojola/project/logs/evaluate_models.out
#SBATCH --requeue

# export MLFLOW_TRACKING_URI="file:///vast/home/jomojola/project/mlruns"

sh /vast/home/jomojola/project/scripts/setup.sh
echo "sparse model eval"
poetry run evaluate_model -mid d9bd74cd -rid 68aa602b71904f658f6c24540dc23d42 -ncls 1
poetry run evaluate_model -mid 19d3d4ef -rid 2088c0ff04634c9682b34883d73c59bc -ncls 1
poetry run evaluate_model -mid 98031c84 -rid 437f892d172e473ca9d7273945da411b -ncls 1
poetry run evaluate_model -mid d6fbc9ca -rid 3f4ce64338084d69a995cf8b1b3fcbd8 -ncls 1
poetry run evaluate_model -mid 1480a0ee -rid 0159461d04c2471d9d501223b64f2e1c -ncls 1
poetry run evaluate_model -mid 10f82008 -rid 69825a38d69049558c47632aec6ef5e4 -ncls 1
poetry run evaluate_model -mid 193eb5fb -rid 8b35e32c4e83422a846412eed76a78d6 -ncls 1
poetry run evaluate_model -mid 5df3e1a8 -rid fd6f780281f946eeaeb6610b8168cab3 -ncls 1
poetry run evaluate_model -mid 0d26b7e8 -rid a0b5f83ef9874fffa2601ee78547e839 -ncls 1
poetry run evaluate_model -mid a1d08723 -rid d20a942c29a94d6dad75903f02e5c28a -ncls 1
poetry run evaluate_model -mid 091219f4 -rid e9268c95642847cbb308d34494d53c94 -ncls 1
poetry run evaluate_model -mid 7c333f0f -rid 5f2f8311fdaf454ba6cfc94846860233 -ncls 1
poetry run evaluate_model -mid ee87ab3c -rid a1242e8494a946bb84f6901042340ecb -ncls 1
poetry run evaluate_model -mid 08111051 -rid 662a6154f590428191844698bafe6cd4 -ncls 1
poetry run evaluate_model -mid 3b10df95 -rid d75b58439dad4e3093faa545ce6f202d -ncls 1
poetry run evaluate_model -mid d1dc3f62 -rid 1bba0db7291c48f387ac379711ca8254 -ncls 1
poetry run evaluate_model -mid 0a45a95b -rid 63d45e3d1e7246d1a93de4730ca317a2 -ncls 1
poetry run evaluate_model -mid 73522049 -rid 0cdc7843328e47838aef72996d99da20 -ncls 1
poetry run evaluate_model -mid 8048abf8 -rid 5b8a289f2fef4e0d8a5119ed09d5ba99 -ncls 1
poetry run evaluate_model -mid a7a40695 -rid 10eb7cbcea96490f9cc925d082c1b3a2 -ncls 1
poetry run evaluate_model -mid c3363698 -rid 1f8ad204388c4639a6b5b75a13567a99 -ncls 1
poetry run evaluate_model -mid d52f212e -rid 494b1463c49d40968e4451bce8cede98 -ncls 1
poetry run evaluate_model -mid c4cd6c98 -rid a7ccc738e6a64252948924b566c516e7 -ncls 1
poetry run evaluate_model -mid fdd2b7e9 -rid 37ad32ea2b1d4c609cce9906c8f09583 -ncls 1
poetry run evaluate_model -mid 137d81c5 -rid 2acf015ba751459f9962fe9882d5ac96 -ncls 1
poetry run evaluate_model -mid 6d562d59 -rid 3658d9b65be6474fbf5b94ee62fcf10a -ncls 1
poetry run evaluate_model -mid 8bda0b23 -rid 0fae0448283e454d9f5ed45b4c137caa -ncls 1
poetry run evaluate_model -mid 1f184878 -rid 6623175718784dd681f47e56d3a14f3b -ncls 1
poetry run evaluate_model -mid 4d0125e9 -rid 9d603f2d27c2434782411369f574c780 -ncls 1
poetry run evaluate_model -mid 41fb05a6 -rid 60be768e2966496091fcaaf5d7850c8f -ncls 1
poetry run evaluate_model -mid 61118432 -rid 6006a7a9a1494f0d8c8e6e40884f7245 -ncls 1
poetry run evaluate_model -mid cfe72a44 -rid 157ef228bd774af883c68488ef86b1f6 -ncls 1
poetry run evaluate_model -mid c5588aae -rid a3723b5d567840ed9d41382f52cbee10 -ncls 1
poetry run evaluate_model -mid 88187cd2 -rid 45cae1de60324a22b6af3788e706df4a -ncls 1
poetry run evaluate_model -mid 88897636 -rid c6e0178120674c2e82dc40735bce04f7 -ncls 1
poetry run evaluate_model -mid 2626c9a2 -rid 30713d731cb543afa49c58cb48c60059 -ncls 1
poetry run evaluate_model -mid e49510ea -rid 60932219a3284a389e8623cdb8d6938d -ncls 1
poetry run evaluate_model -mid d6368fec -rid 7d614bcc539048b1a7d75f7d42308b9f -ncls 1
poetry run evaluate_model -mid 6db54424 -rid 04cc8ea1cc1b49efaf87db2c1562cc1b -ncls 1
poetry run evaluate_model -mid 50807743 -rid f0d1c13e37d049dea167ff0cd244e1c2 -ncls 1
poetry run evaluate_model -mid f88313b0 -rid e728a52f73304b618964a9b52ede9d2a -ncls 1
poetry run evaluate_model -mid 201620b2 -rid b52c060ba90d4ae4995a849a2cb9b42c -ncls 1
poetry run evaluate_model -mid 1cb2b051 -rid 387f2aae241a4d6596a6a05358c877fe -ncls 1
poetry run evaluate_model -mid b25b3ead -rid f8fdb01d63ba48978b67c0bb6d9cf192 -ncls 1
poetry run evaluate_model -mid 78e67e39 -rid 592210bee340463f950dc2c578675025 -ncls 1
poetry run evaluate_model -mid f3ded9f1 -rid ba2af17403b64a5eb73b17b30941ebfc -ncls 1
poetry run evaluate_model -mid 25ce2dcb -rid 37e3b495d7ab4161b0727d2fb3a57124 -ncls 1
poetry run evaluate_model -mid 5277bc7e -rid 05961bb58c8a40db8f818e59a1900d10 -ncls 1
poetry run evaluate_model -mid 0aef6f63 -rid b395e659bde944edbecbf34205c6e4e2 -ncls 1
poetry run evaluate_model -mid 7609af79 -rid b34dda20bd02432b891121a7e6bcdc76 -ncls 1
poetry run evaluate_model -mid 0304bee6 -rid 173fc196afb7433795688b1f8f952a62 -ncls 1
poetry run evaluate_model -mid 2d2db52f -rid f399cbeabc8740b0abfdd370dba8a7a5 -ncls 1
poetry run evaluate_model -mid 7fe7beaa -rid ba8668cef7b14e2b9b1ed6f3615d2fb5 -ncls 1
poetry run evaluate_model -mid e44148f8 -rid a8a03570ae6d45f68269b29c2ada18d6 -ncls 1
poetry run evaluate_model -mid a8349da4 -rid 96b0290ffecb452ca97a0cf1843b58cb -ncls 1
poetry run evaluate_model -mid 2367277d -rid 9eee421276bc42bb8895076b142ef7c8 -ncls 1
poetry run evaluate_model -mid 699bed83 -rid a504c597bff447118487fa932d94b8a4 -ncls 1
poetry run evaluate_model -mid 0752911a -rid a60839dcde3a401d8bd34685908359d1 -ncls 1
poetry run evaluate_model -mid 8d94e2e9 -rid 58484b3d49db4f1180c78c345f99cf29 -ncls 1
poetry run evaluate_model -mid 95762bd9 -rid efd5d06880864d65a20bf68b85e4c2c4 -ncls 1
poetry run evaluate_model -mid 943ae619 -rid f32e85c4316a4138a904e38db7d45db1 -ncls 1
poetry run evaluate_model -mid 93dc5689 -rid e0670352b924409488a61791891c67c2 -ncls 1
poetry run evaluate_model -mid 691d7ea9 -rid 0dd40b89c9c843e1a9f15b99d10c4cea -ncls 1
poetry run evaluate_model -mid 1f665244 -rid 89c4c356a4834a2cbea7973bbeb7ccef -ncls 1
poetry run evaluate_model -mid f0605033 -rid 082eeb2ee48e4289876890e6222518b7 -ncls 1
poetry run evaluate_model -mid daa82453 -rid 9ced12388a3f4c0b8986fb2c7b63baa7 -ncls 1
poetry run evaluate_model -mid 0f54124f -rid 4b9731567282446ba233bb8236a55637 -ncls 1
poetry run evaluate_model -mid 5ac3c31a -rid 6bcdbe59decc4389a45d1eb9a93b5f3b -ncls 1
poetry run evaluate_model -mid c885f74f -rid f7cf28915e60421fb991c9ef69e22d31 -ncls 1
poetry run evaluate_model -mid 300b2f61 -rid 7fee4bd3978b49dfb0c4ab1af68ceb81 -ncls 1
poetry run evaluate_model -mid 4c62f344 -rid 266f3f67760343e3b16c04e3a300a57b -ncls 1
poetry run evaluate_model -mid 867b015d -rid 724f7c9a56674624a83cd6b266e0fa9b -ncls 1
poetry run evaluate_model -mid 92f4e98c -rid 16005c47ac974a658ae203c36695cda4 -ncls 1
poetry run evaluate_model -mid cc345f33 -rid 92b0cff75cf8479e87835bccf6b5f36b -ncls 1
poetry run evaluate_model -mid 95a08fc0 -rid 428f516bdbb24033a1ddd593b5fd7f01 -ncls 1
poetry run evaluate_model -mid b308e37a -rid 24901e091d5b4e58a45f701a2e2ef8b6 -ncls 1
poetry run evaluate_model -mid 6c317bc9 -rid fe22033f55544a05b1256ad745914fc1 -ncls 1


echo "random model eval"
poetry run evaluate_model -mid 8c69c464 -rid 602bb8b0bdd84e868789a962113c7729 -ncls 1 -r
poetry run evaluate_model -mid f8b6fc3a -rid 4fb65866bcc54b59bbd44faa1ea0ea87 -ncls 1 -r
poetry run evaluate_model -mid c6d4d288 -rid 2d2b7a0c2e324ebb8455da214292ce71 -ncls 1 -r
poetry run evaluate_model -mid 79cb833e -rid 68a21a1eab8f4b5294d19c6b91c4c602 -ncls 1 -r
poetry run evaluate_model -mid 1a26b974 -rid 6ecfc7e2c4c846fea9e0043fd7f679c9 -ncls 1 -r
poetry run evaluate_model -mid 42e9afab -rid fb5288e4aec54e14b27b2e693780b315 -ncls 1 -r
poetry run evaluate_model -mid d0da20ec -rid 338f837749ea4669a5033a51d6d58abd -ncls 1 -r
poetry run evaluate_model -mid 486eabf6 -rid ec339407044a444daccec1d745cd1967 -ncls 1 -r
poetry run evaluate_model -mid e6abe84b -rid bd4b1cc57acc40caa2e9f2b7c020638e -ncls 1 -r
poetry run evaluate_model -mid d6a3a1c7 -rid 94b3d394564f45a1a6512d0f1d0d42b8 -ncls 1 -r
poetry run evaluate_model -mid 1bde61cc -rid 26009589a3b24f1880d57d62517d34fd -ncls 1 -r
poetry run evaluate_model -mid 25da2d9d -rid b6431cf431534e3eacaf341de98f5a37 -ncls 1 -r
poetry run evaluate_model -mid 6f81c36c -rid a46353abd5d449ef8e0baa3a0cf4d14d -ncls 1 -r
poetry run evaluate_model -mid 1d43520b -rid 5fe8f9e5df2b45e8a1b750c0e6fb7298 -ncls 1 -r
poetry run evaluate_model -mid f328009b -rid 1f54d616dd5a45629d00f8570df36b74 -ncls 1 -r
poetry run evaluate_model -mid 9bb9d447 -rid 46c76145013f4523ac78a30f91cbbd58 -ncls 1 -r
poetry run evaluate_model -mid c1f19962 -rid 0b74a6b5e2f94759a9a781ac53f84147 -ncls 1 -r
poetry run evaluate_model -mid 082e9f5a -rid 8d7f06e0dd5d4541bf69d49d6dd46f00 -ncls 1 -r
poetry run evaluate_model -mid b68dc10a -rid 0511ff9db13d4f53a8f60f9f434d5b3e -ncls 1 -r
poetry run evaluate_model -mid 6c61c433 -rid c305f8cb59f448a8a5f77521674afcec -ncls 1 -r
poetry run evaluate_model -mid 28da5211 -rid 16d6b18e30824246bf1632be81bee30c -ncls 1 -r
poetry run evaluate_model -mid c6682d15 -rid fa14d07bd9d54357a77793a4d4f6f83c -ncls 1 -r
poetry run evaluate_model -mid 5b41d5a6 -rid 6b3f8f22711548e8bdf3f9272e67d29c -ncls 1 -r
poetry run evaluate_model -mid 41842cba -rid dc6ebbc0f48149f3bab9c0143ea11602 -ncls 1 -r
poetry run evaluate_model -mid b8c724b3 -rid 23a129c647e14b6187063852174b7bc9 -ncls 1 -r
poetry run evaluate_model -mid 9eec2245 -rid 320b66bbe2b14e14a7273ffc6dc08600 -ncls 1 -r
poetry run evaluate_model -mid f1408192 -rid 648d930ff0db40efbe90c974b5d18d31 -ncls 1 -r
poetry run evaluate_model -mid 3c90e5b9 -rid ca5c8015848343408027ca90e27eee27 -ncls 1 -r
poetry run evaluate_model -mid 0110818c -rid 329579f4e79a4df0b08251181c056ff4 -ncls 1 -r
poetry run evaluate_model -mid 3b72ecff -rid 614b65fe729b432f9cd609b587c3ba3d -ncls 1 -r
poetry run evaluate_model -mid 4da32722 -rid 9d2af3dad90d41bcaee73acfe0fa3418 -ncls 1 -r
poetry run evaluate_model -mid fd03c220 -rid 2490c96489d747c3a0921db6d1ce3c62 -ncls 1 -r
poetry run evaluate_model -mid 280db935 -rid 5532cd54c448444e8cce7e52e0b6c55e -ncls 1 -r
poetry run evaluate_model -mid f3429f70 -rid 396d9fcd5d8e4aa3b42c02bcbcf73cb6 -ncls 1 -r
poetry run evaluate_model -mid 5cefe561 -rid 6715483861a64d44a4deae28f7e6e193 -ncls 1 -r
poetry run evaluate_model -mid 372c313b -rid 52a227aaab9e4683bfeab32e20926c1d -ncls 1 -r
poetry run evaluate_model -mid 7ad7d288 -rid e97c9a6d687540789d08d8c475ca1018 -ncls 1 -r
poetry run evaluate_model -mid 86ae722b -rid ddb5e2723a014ca3915ba644c15af65f -ncls 1 -r
poetry run evaluate_model -mid fa3338b2 -rid 731c6458ce9b4876965da5d9ebdf984c -ncls 1 -r
poetry run evaluate_model -mid 21dbfb7b -rid 993230c23edf44838119e9026484ce0f -ncls 1 -r
poetry run evaluate_model -mid b1939e69 -rid 5c8b54f9683547b0890c0fbf64bdfe53 -ncls 1 -r
poetry run evaluate_model -mid 151e9e24 -rid 195a0ab55e25430091f4f8bf63727820 -ncls 1 -r
poetry run evaluate_model -mid 3c79a11b -rid 159d400b29254650ab0c395e0a986cbe -ncls 1 -r
poetry run evaluate_model -mid ebf5051a -rid 6408df90c6db4bafb41dfbc4e9f91a70 -ncls 1 -r
poetry run evaluate_model -mid 748ca054 -rid 20afc7b9f34041abb13799c184842cf0 -ncls 1 -r
poetry run evaluate_model -mid 62a272eb -rid d3d16ecbcabe449e9cf087fbef4666b8 -ncls 1 -r
poetry run evaluate_model -mid 2297cd26 -rid 0d659052362e4d0eb3e429f39a351541 -ncls 1 -r
poetry run evaluate_model -mid 2f32e6dd -rid 8641331812454265882852b2ab09521a -ncls 1 -r
poetry run evaluate_model -mid 86dc813a -rid a9b28eb6fee245019f3b2e6eac78d361 -ncls 1 -r
poetry run evaluate_model -mid 39c1b808 -rid 2cf9c6fa9a7d4217a1ad57a682e6136e -ncls 1 -r
poetry run evaluate_model -mid 16f496fd -rid 373330e369db4dbdac0acefcc768ef29 -ncls 1 -r
poetry run evaluate_model -mid 69e0758e -rid a32458afd6e94fc7833be6e4389c44b7 -ncls 1 -r
poetry run evaluate_model -mid 8160a4e8 -rid 5d9baff35d964876844379e3ef8f184c -ncls 1 -r
poetry run evaluate_model -mid a1936513 -rid a70d0a5a6777490b854e23002372ccca -ncls 1 -r
poetry run evaluate_model -mid a1e83c54 -rid 9e821290b20f49f08e61b76fb03ae198 -ncls 1 -r
poetry run evaluate_model -mid ce555b66 -rid 632a5f3e62f64f3c9c64e5ed6e00102e -ncls 1 -r
poetry run evaluate_model -mid 30eb7c99 -rid 5dafdbfb9c2143c5826c509e37cd28a1 -ncls 1 -r
poetry run evaluate_model -mid 393e133d -rid 31ae73e3116b4e0cb5d82ac60c9abbaa -ncls 1 -r
poetry run evaluate_model -mid c11baca9 -rid a4c3f7013fc845e5a09bfef43510d687 -ncls 1 -r
poetry run evaluate_model -mid c4769f2d -rid 23a2a2f6504a463f91dc9ff6df520527 -ncls 1 -r
poetry run evaluate_model -mid fc2682eb -rid 98344abf832c451cac1127b55f86dffa -ncls 1 -r
poetry run evaluate_model -mid 3123b4c2 -rid af1f860385394367a69c9a714806ec44 -ncls 1 -r
poetry run evaluate_model -mid a99a1bd0 -rid a80787580d3f40739869a22b19b16185 -ncls 1 -r
poetry run evaluate_model -mid d1e5106d -rid d051f5b7ff5b4c919aa33c1ee0efe587 -ncls 1 -r
poetry run evaluate_model -mid 142915f1 -rid 5d9a47ff12794d58bd662a82bff6bbb5 -ncls 1 -r
poetry run evaluate_model -mid 5b471ee9 -rid 47645ca7cb0d4364a6d1b371cf939b62 -ncls 1 -r
poetry run evaluate_model -mid 7e16c500 -rid 0887faeb8f8b44299381566b5f7925fc -ncls 1 -r
poetry run evaluate_model -mid 09fab0b7 -rid 0d1594d2f09947768897c6c875a19029 -ncls 1 -r
poetry run evaluate_model -mid 825c1cc0 -rid 2d5796e2c057474c95aa25b01a4ab38d -ncls 1 -r
poetry run evaluate_model -mid 85ba1367 -rid 3d4d3bfc9a0b494aab0c74d6c5437da9 -ncls 1 -r
poetry run evaluate_model -mid 2d4cbcf1 -rid e80a906e9c284d14b4b78c28ffea7546 -ncls 1 -r
poetry run evaluate_model -mid 30040981 -rid 457c0355faaf48d5ae052d382277ca78 -ncls 1 -r
poetry run evaluate_model -mid a27d5e9f -rid a3e04e6e0f8d4140a64937a4173eacf9 -ncls 1 -r
poetry run evaluate_model -mid f5876c9a -rid 549234c73ae747129eaa72162dd9b972 -ncls 1 -r
poetry run evaluate_model -mid bc52fd3e -rid 26109527e52a40fe845b83fd6d97e69a -ncls 1 -r
poetry run evaluate_model -mid 3b797e2c -rid 66243af0219c4058ba727fc3a44caf2b -ncls 1 -r
poetry run evaluate_model -mid d883ce10 -rid 66718d6f08794c739abfc53163dfd013 -ncls 1 -r
poetry run evaluate_model -mid fbb5caa3 -rid 58aa7f9b91444a2ba4fdc3b71bf231cf -ncls 1 -r
poetry run evaluate_model -mid fc98d1ac -rid d30912513e1b4592a70930db30a81a74 -ncls 1 -r
poetry run evaluate_model -mid 10907efd -rid 98b0ce71fb634be99d230fd00643985c -ncls 1 -r
poetry run evaluate_model -mid 8dcc32b3 -rid b3c99d423af34582938809c7233b4e06 -ncls 1 -r
poetry run evaluate_model -mid ae558cdd -rid 52b216368f864601ad622068105b0efd -ncls 1 -r
poetry run evaluate_model -mid a239dbde -rid b39dfb8d24364191a24d8d48ecc273fa -ncls 1 -r
poetry run evaluate_model -mid 83b2efae -rid 871777db2e454e8189d7d8fe784a2c2c -ncls 1 -r
poetry run evaluate_model -mid eeeb36a9 -rid 2daea68da7da4f0c8d1e3b18edeeb2e2 -ncls 1 -r
poetry run evaluate_model -mid df08b653 -rid 2f6e6b35fcbc42b7b874e12ab35effe5 -ncls 1 -r
poetry run evaluate_model -mid 36e01cec -rid 71ce4f7e623d48cf85f8b5a1ed766063 -ncls 1 -r
poetry run evaluate_model -mid 3d3d4115 -rid 8be32e9b22544ef285319f40a91381f4 -ncls 1 -r
poetry run evaluate_model -mid 34b097f4 -rid 80b4a0c5383f43eaafdc3d6c7572b5a9 -ncls 1 -r
poetry run evaluate_model -mid 75095f52 -rid fff0b4a4e5794edeaa4dac9b7db0ffd4 -ncls 1 -r
poetry run evaluate_model -mid 24c08233 -rid 96ac0e19f5f6446789810206d9eeadb8 -ncls 1 -r
poetry run evaluate_model -mid d42f3c28 -rid d838733445544c349106544183c3266b -ncls 1 -r
poetry run evaluate_model -mid 4998fc8d -rid e3b2139e1e354d7a9aadf27e443d3544 -ncls 1 -r
poetry run evaluate_model -mid 6a6850f6 -rid 3bbea446a03f44df979d9a6b814649e5 -ncls 1 -r
poetry run evaluate_model -mid 574d30d7 -rid 1752c787c5cb4e5182a43f098884445c -ncls 1 -r
poetry run evaluate_model -mid 6056534c -rid 677e081ce76644e2b9b58542104ded5d -ncls 1 -r
poetry run evaluate_model -mid 9fd68e06 -rid 535ac2af89144698bc0ec07ca49582b5 -ncls 1 -r
poetry run evaluate_model -mid 16950700 -rid 2bda6bd0dc5f4297a31034f5f49624d9 -ncls 1 -r
poetry run evaluate_model -mid e65ce341 -rid 71771aa3048b46fbb8a70091c5c5926c -ncls 1 -r
poetry run evaluate_model -mid cc2f400d -rid d995928741a341639d31bb3a5a4a9009 -ncls 1 -r
poetry run evaluate_model -mid 3d56bd2d -rid 2710ad5239584e0a915ea9111850357c -ncls 1 -r
poetry run evaluate_model -mid fc3e3920 -rid 8482003ac4dd4c86b785a83485e45d4e -ncls 1 -r
poetry run evaluate_model -mid 40759172 -rid 3e03f465b79840dd90a776ac66d88a12 -ncls 1 -r
poetry run evaluate_model -mid 5ff016d3 -rid 431fe66410aa412d8e52a26bb15e02cf -ncls 1 -r
poetry run evaluate_model -mid 55d9d7e0 -rid 144b447e716548b2bd71a96180e2215a -ncls 1 -r
poetry run evaluate_model -mid ac78132f -rid 93a19c077a4346429e3c36de4bc10949 -ncls 1 -r
poetry run evaluate_model -mid 71237bb3 -rid 6cd6a0c95afb4347b55c0d3046cbdf4c -ncls 1 -r
poetry run evaluate_model -mid 0878d1ab -rid c1c2de6d9dbf474d86692a0be68f68a7 -ncls 1 -r
poetry run evaluate_model -mid b2dda203 -rid 5f50597a11584d4789fcf2667e1e771a -ncls 1 -r
poetry run evaluate_model -mid 61b066f4 -rid 2cc9c5aa6ad54dc2a4dd56ab4a730fc2 -ncls 1 -r
poetry run evaluate_model -mid 74beb110 -rid 223ed84c32804921ae311f8a3c99ff17 -ncls 1 -r
poetry run evaluate_model -mid 0e2ba3e4 -rid 1c183173ad5446b6b49a09247418a828 -ncls 1 -r
poetry run evaluate_model -mid bc63a05c -rid 00324e26b254483fb1e7a7491c405ed7 -ncls 1 -r
poetry run evaluate_model -mid 9f913164 -rid ebd7c5e9a19645bda75dc2d23fd80288 -ncls 1 -r
poetry run evaluate_model -mid 89df7254 -rid e45d019d00a04ff2815ecbee3ef5dd67 -ncls 1 -r
poetry run evaluate_model -mid 24347cac -rid bb5821a62ca54b79944126f0440019d6 -ncls 1 -r
poetry run evaluate_model -mid 41b9ebb4 -rid cf69139c18324d3c98cc0838d30f02f7 -ncls 1 -r
poetry run evaluate_model -mid 35eae2de -rid 369b9e2d957c4ed5bdd72348d8aee762 -ncls 1 -r
poetry run evaluate_model -mid 82d9f461 -rid edba2aadbb354952bbadadc63e7925b3 -ncls 1 -r