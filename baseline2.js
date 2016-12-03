var n1 = 10;
var n2 = 10;
var n3 = 10;
//var k1=0; var k2=10;
// var data = {
//     trial1: [0, 10],
//     trial2: [5, 5],
//     trial3: [3, 7],
//     trial4: [0, 5],
//     trial0: [2, 4],
//     trial5: [1, 10],
//     trial6: [4, 3],
//     trial7: [2, 3],
//     trial8: [2, 2],
//     trial9: [1, 4]
// };

var data = {
    trial0: [0, 1],
    trial1: [1, 2],
    trial2: [2, 3],
    trial3: [3, 4],
    trial4: [4, 5],
    trial5: [0, 2],
    trial6: [1, 3],
    trial7: [2, 4],
    trial8: [3, 5],
    trial9: [0, 3],
    trial10: [0, 3],
    trial11: [1, 4],
    trial12: [2, 5]
};
var trialNames = ['trial0', "trial1", "trial2", "trial3", "trial4", "trial5", "trial6",
    "trial7", "trial8", "trial9", 'trial10', 'trial11', 'trial12'
];

var l1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

var marginalize = function(dist, key) {
    return Infer({
        method: "enumerate"
    }, function() {
        return sample(dist)[key];
    });
};

var runExperiment = function(opts, datum, i, j, k) {
    return Infer(opts, function() {
        var p = [uniform(0, 1), uniform(0, 1), uniform(0,1)];
        observe(Binomial({
            p: p[i],
            n: n1
        }), datum[0]);
        observe(Binomial({
            p: p[j],
            n: n2
        }), datum[1]);
        observe(Binomial({
            p: p[k],
            n: n3
        }), datum[2]);
        var pred1 = binomial(p[i], n1);
        var pred2 = binomial(p[j], n2);
        var pred3 = binomial(p[k], n3);
        return {
            'p1': p[i],
            'p2': p[j],
            'p3': p[k],
            pred1: pred1,
            pred2: pred2,
            pred3: pred3
        };
    });
};

var getExperiment = function(h, datum) {
    var opts_inner = {
        method: 'MCMC',
        samples: 20000,
        burn: 30000
    };
    if (h == "0 0 0") {
        return runExperiment(opts_inner, datum, 0, 0, 0);
    } else if (h == "0 0 1") {
        return runExperiment(opts_inner, datum, 0, 0, 1);
    } else if (h == "0 1 2") {
        return runExperiment(opts_inner, datum, 0, 1, 2);
    } else if (h == "0 1 0") {
        return runExperiment(opts_inner, datum, 0, 1, 0);
    } else if (h == "1 0 0") {
        return runExperiment(opts_inner, datum, 1, 0, 0);
    }
};

var study = function(datum) {
    return Infer({
        method: 'enumerate'
    }, function() {
        // var hypothesis = flip() ? 'H0' : 'HA';
        var hypotheses = ['0 0 0', '0 0 1', '0 1 2', "0 1 0", "1 0 0"];
        var hypothesis = hypotheses[sample(RandomInteger({n: hypotheses.length}))];
        var experiment = getExperiment(hypothesis, datum);
        var m_pred1 = marginalize(experiment, 'pred1');
        var m_pred2 = marginalize(experiment, 'pred2');
        var m_pred3 = marginalize(experiment, 'pred3');
        observe(m_pred1, datum[0]);
        observe(m_pred2, datum[1]);
        observe(m_pred3, datum[2]);
        return hypothesis;
    });
};

var visualizeStudy = function(datum) {
    print("H0 vs HA probabilities based on data=[" + datum + ']:');
    var s = study(datum);
    viz.auto(s);
    //print("probabilities p1, p2: "+mean(s[1])+', '+mean(s[2]))
    return s;
};
visualizeStudy([5, 5, 5]);
visualizeStudy([0, 0, 10]);
visualizeStudy([0, 5, 10]);
// map(function(i,j) {
//   map(function(j) {
//     if (i>j) return;
//     print("Running program on data = "+i+", "+j+". Probability of H0:")
//     var s = study([i,j])
//     print(Math.exp(s.score("H0")))
//   }, l1)
// }, l1)
