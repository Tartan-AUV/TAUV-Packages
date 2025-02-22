#include "global_map.h"
#include <cmath>

using namespace std;

#define NO_PREDICTED_NUM -1
#define MIN_DETECTIONS 5
#define DOUBLE_ONE 1.1

bool unknownTag(Detection &det){return det.tag.compare(UNKNOWN_KEYWORD)==0;}

TrackerMaster::TrackerMaster(string tag, ros::NodeHandle& handler)
{
    nodeHandler = handler;

    TrackerList= {};
    Zombies = {};
    needsUpdating = false;
    numDetections = 0;
    max_threshold = 0;
    feature_tag = tag;

    predictedTrackerNum = getParam("expected_count", NO_PREDICTED_NUM);
    DECAY_THRESHOLD = getParam("DECAY_THRESHOLD", 100);
    recency_weight = getParam("recency_weight");
    frequency_weight = getParam("frequency_weight");
}

double TrackerMaster::getParam(string property, double def)
{
    double val;

    //try tag-specific
    if(!ros::param::get("tag_params/"+feature_tag+"/"+property, val)){
        //try default tag
        if(!ros::param::get("tag_params/"+property, val)){
            return def;
        }
    }

    return val;
}

shared_ptr<Tracker> TrackerMaster::getTracker(size_t featureIdx)
{
    return TrackerList[featureIdx];
}

vector<shared_ptr<Tracker>> TrackerMaster::getTrackers(){return TrackerList;}

size_t TrackerMaster::getNumTrackers(){return TrackerList.size() - Zombies.size();}

//this will delete Zombies if they are taking up more than half of the Tracker List array
void TrackerMaster::makeUpdates()
{
    needsUpdating = false;

    //122 moment
    if(Zombies.size()==0 || Zombies.size()<TrackerList.size()/2){return;}

    size_t del = Zombies.top();
    Zombies.pop();

    size_t curIdx = del;

    for(size_t featureIdx = del; featureIdx < TrackerList.size(); featureIdx++)
    {
        if(featureIdx<del){
            TrackerList[curIdx]=getTracker(featureIdx);
            curIdx++;
        }

        else{
            //delete TrackerList[del];

            if(Zombies.size()==0){
                del=TrackerList.size();
            }
            else{
                del = Zombies.top();
                Zombies.pop();
            }
        }
    }

    TrackerList.erase(TrackerList.begin()+curIdx, TrackerList.end());
}

double TrackerMaster::getMaxThreshold(){return max_threshold;}

bool TrackerMaster::reassignable(size_t featureIdx, Detection &detection)
{
    return getTracker(featureIdx)->reassignable(detection);
}

shared_ptr<Tracker> TrackerMaster::popReassignable(size_t featureIdx, Detection& det)
{
    shared_ptr<Tracker> F = getTracker(featureIdx)->makeReassignment(det);
    deleteTracker(featureIdx);

    return F;
}

void TrackerMaster::addTracker(Detection &detection)
{
    if(Zombies.size()>0)
    {
        int featureIdx = Zombies.top();
        Zombies.pop();

        getTracker(featureIdx)->reinit(detection);
        numDetections+=1;
    }
    else
    {
        shared_ptr<Tracker> NEW (new Tracker(detection, nodeHandler));
        addTracker(NEW);
    }
}

void TrackerMaster::addTracker(shared_ptr<Tracker> F)
{
    numDetections+=F->getNumDetections();
    TrackerList.push_back(F);
    max_threshold = max(max_threshold, F->getThreshold());
}

void TrackerMaster::deleteTracker(size_t featureIdx)
{
    shared_ptr<Tracker> F = getTracker(featureIdx);
    numDetections -= F->getNumDetections();

    F->reset();
    Zombies.push(featureIdx);
    needsUpdating = true;
}

double TrackerMaster::getDecay(size_t featureIdx, int totalDetections)
{
    double recDecay = recency_weight*getTracker(featureIdx)->recencyCalc(numDetections);
    double freqDecay = frequency_weight*getTracker(featureIdx)->frequencyCalc(totalDetections);
    double decay = (recDecay+freqDecay)/(frequency_weight+recency_weight);

    // cout<<feature_tag<<"\n";
    // cout<<"decay: "<<decay<<"\n\n";

    return decay;
}

double TrackerMaster::decay(size_t featureIdx, int totalDetections)
{
    shared_ptr<Tracker> F = getTracker(featureIdx);
    F->incrementRecency();

    return totalDetections>MIN_DETECTIONS && (F->State==POTENTIAL_ZOMBIE) && getDecay(featureIdx, totalDetections)<DECAY_THRESHOLD;
}

void TrackerMaster::addDetection(size_t featureIdx, Detection &detection)
{
    numDetections+=1;
    getTracker(featureIdx)->addDetection(detection);
}

vector<double> TrackerMaster::getSimilarityRow(shared_ptr<Tracker> F, vector<Detection> &detections)
{
    vector<double> featureSimMatrix(detections.size());

    for(size_t i=0; i<detections.size(); i++)
    {
        Detection det = detections[i];
        featureSimMatrix[i] = F->getSimilarityCost(det, numDetections);
    }

    return featureSimMatrix;
}

int TrackerMaster::generateSimilarityMatrix(vector<Detection> &detections, vector<vector<double>> &costMatrix, size_t trackerNum, vector<pair<shared_ptr<TrackerMaster>, int>> &trackerList)
{
    //make any needed updates prior to matching
    if(needsUpdating){makeUpdates();}

    size_t featureNum = trackerNum;
    for(size_t featureIdx = 0; featureIdx<TrackerList.size(); featureIdx++)
    {
        shared_ptr<Tracker> F = getTracker(featureIdx);
        if(F->State==ZOMBIE){continue;}

        costMatrix[featureNum] = getSimilarityRow(F, detections);
        trackerList[featureNum] = make_pair(shared_from_this(), featureIdx);
        featureNum++;
    }

    return featureNum;
}

bool TrackerMaster::validCost(size_t featureIdx, double cost)
{
    int oversaturated = predictedTrackerNum!=NO_PREDICTED_NUM && (predictedTrackerNum<=int(TrackerList.size()));

    //increase threshold for tracker creation if too many trackers
    return getTracker(featureIdx)->validCost(cost, oversaturated);
}

Tracker::Tracker(Detection &initial_detection, ros::NodeHandle& handler, shared_ptr<KalmanFilter> position, shared_ptr<KalmanFilter> orientation)
{
    initialize(initial_detection, handler);

    kPosition = position;
    kOrientation = orientation;
}

Tracker::Tracker(Detection &initial_detection, ros::NodeHandle& handler) :
kPosition (new KalmanFilter((initial_detection.tag+"/position"), initial_detection.position, initial_detection.confidence)),
kOrientation (new KalmanFilter((initial_detection.tag+"/orientation"), initial_detection.orientation, initial_detection.confidence))
{
    initialize(initial_detection, handler);
}

void Tracker::initialize(Detection &initial_detection, ros::NodeHandle& handler)
{
    feature_tag = initial_detection.tag;
    tracker_type = initial_detection.tracker_type;
    State = ACTIVE;

    num_detections = 1;
    recency = 1;
    SE2 = initial_detection.SE2;

    readParams();

    nodeHandler = handler;
    decayTimer = handler.createTimer(ros::Duration(min_decay_time), &Tracker::setPotentialZombie, this, true);
}

bool Tracker::reassignable(Detection& det)
{
    return reassignable_tracker && (diffTag(det.tag) || diffTracker(det.tracker_type));
}

//were essentially copying ourselves
shared_ptr<Tracker> Tracker::makeReassignment(Detection &detection)
{
    shared_ptr<KalmanFilter> POS = kPosition->copy(detection.tag);
    shared_ptr<KalmanFilter> OR = kOrientation->copy(detection.tag);
    shared_ptr<Tracker> F (new Tracker (detection, nodeHandler, POS, OR));

    F->setNumDetections(num_detections);
    return F;
}

void Tracker::readParams()
{
    //init params TODO: add defaults for each
    mahalanobis_threshold = getParam("mahalanobis_threshold");
    tag_weight = getParam("matching_weights/tag_weight");
    distance_weight = getParam("matching_weights/distance_weight");
    orientation_weight = getParam("matching_weights/orientation_weight");
    recency_matching_weight = getParam("matching_weights/recency_weight");
    frequency_matching_weight = getParam("matching_weights/frequency_weight");
    confidence_weight = getParam("matching_weights/confidence_weight");
    oversaturation_penalty = getParam("matching_weights/oversaturation_penalty");
    min_decay_time = getParam("min_decay_time");
    tracker_bias = getParam("tracker_bias");
    reassignable_tracker = getParam("reassignable", false);
}

void Tracker::resetTimer()
{
    decayTimer.setPeriod(ros::Duration(min_decay_time), true);
    decayTimer.start();
}

double Tracker::getParam(string property, double def)
{
    double val;

    //try tracker-specific, tag-specific
    if(!ros::param::get("tracker_params/"+tracker_type+"/tag_params/"+feature_tag+"/"+property, val)){
        //try tracker-specific, default tag
        if(!ros::param::get("tracker_params/"+tracker_type+"/"+property, val)){
            //try default tracker, tag-specific tag
            if(!ros::param::get("tracker_params/default/tag_params/"+feature_tag+"/"+property, val))
            {
            //try default tracker, default tag settings
                if(!ros::param::get("tracker_params/default/"+property, val))
                {
                    return def;
                }
            }
        }
    }

    return val;
}

bool Tracker::getParam(string property, bool def)
{
    bool val;

    //try tracker-specific, tag-specific
    if(!ros::param::get("tracker_params/"+tracker_type+"/tag_params/"+feature_tag+"/"+property, val)){
        //try tracker-specific, default tag
        if(!ros::param::get("tracker_params/"+tracker_type+"/"+property, val)){
            //try default tracker, tag-specific tag
            if(!ros::param::get("tracker_params/default/tag_params/"+feature_tag+"/"+property, val))
            {
            //try default tracker, default tag settings
                if(!ros::param::get("tracker_params/default/"+property, val))
                {
                    return def;
                }
            }
        }
    }

    return val;
}

void Tracker::reset()
{
    State = ZOMBIE;
    num_detections = 0;
    decayTimer.stop();
}

void Tracker::reinit(Detection &initial_detection)
{
    State = ACTIVE;

    kPosition->reset(initial_detection.position, initial_detection.confidence);
    kOrientation->reset(initial_detection.orientation, initial_detection.confidence);

    num_detections = 1;

    tracker_type = initial_detection.tracker_type;
    readParams();
}

void Tracker::setPotentialZombie(const ros::TimerEvent& event)
{
    State = POTENTIAL_ZOMBIE;
    decayTimer.stop();
}

Eigen::Vector3d Tracker::getPosition()
{
    return kPosition->getEstimate();
}

Eigen::Vector3d Tracker::getOrientation()
{
    return kOrientation->getEstimate();
}

string Tracker::getTag()
{
    return feature_tag;
}

string Tracker::getTrackerType()
{
    return tracker_type;
}

double Tracker::getConfidence()
{
    return kOrientation->getConfidence();
}

size_t Tracker::getNumDetections()
{
    return num_detections;
}

bool Tracker::is_SE2(){return SE2;}

void Tracker::incrementRecency(){recency++;}

void Tracker::setNumDetections(int num){num_detections = num;}

void Tracker::addDetection(Detection& detection)
{
    if(State==ZOMBIE){
        reinit(detection);
    }
    else{
        State = ACTIVE; //could be POTENTIAL_ZOMBIE

        num_detections++;
        recency = 1;

        //dimensionality of data
        size_t dim_pos = detection.SE2 ? 2 : 3;
        size_t dim_or = detection.SE2 ? 1 : 3;

        kPosition->updateEstimate(detection.position, dim_pos, detection.confidence);
        kOrientation->updateEstimate(detection.orientation, dim_or, detection.confidence);
        SE2 = detection.SE2 && SE2;
    }

    resetTimer();
}

double Tracker::getDistance(Detection& detection)
{
    //if either the tracker is 2D or detection is 2D, we ignore the last distance
    if(SE2 || detection.SE2){detection.position[2] = getPosition()[2];}
    return (getPosition()-detection.position).norm();
}

double Tracker::getRotation(Detection& detection)
{
    //only compare yaw if SE2!
    if(SE2 || detection.SE2){detection.orientation[2] = getOrientation()[2]; detection.orientation[1] = getOrientation()[1];}
    return (getOrientation()-detection.orientation).norm();
}

//true if tags are different, false if same
bool Tracker::diffTag(string det_tag)
{
    return det_tag.compare(feature_tag)!=0;
}

bool Tracker::diffTracker(string det_tracker_type)
{
    return det_tracker_type.compare(tracker_type)!=0;
}

double Tracker::getSimilarityCost(Detection &det, size_t total_num_detections)
{
    //feature-based similarity costs
    double distance = distance_weight*getDistance(det);
    double orientation = orientation_weight*getRotation(det);

    //detection tags match with exact tag matches or unknown tags
    double tagDist = tag_weight*(!unknownTag(det) && diffTag(det.tag));

    // cout<<feature_tag<<"\n";
    // cout<<det.tag<<"\n";
    // cout<<tagDist<<"\n";

    //tracker-bias similarity costs
    double freqDist = frequency_matching_weight*(DOUBLE_ONE-(((double)num_detections)/total_num_detections));
    double recDist = recency_matching_weight*(recency/total_num_detections);
    //trackers we are less confident in are less likely to be matched to
    double confidence_bias = confidence_weight*(DOUBLE_ONE-getConfidence());
    // cout<<tag<<"\n";

    // cout<<"dist: "<<distance+orientation+tagDist+freqDist+recDist+confidence_bias<<"\n\n";

    return distance+orientation+tagDist+freqDist+recDist+confidence_bias+tracker_bias;//total_weight;
}

bool Tracker::validCost(double cost, bool oversaturated)
{
    double thresh = (mahalanobis_threshold+(oversaturation_penalty*((int)oversaturated)));

    // cout<<"thresh: "<<thresh<<"\n";
    // cout<<"cost: "<<cost<<"\n";
    // cout<<"mahal: "<<mahalanobis_threshold<<"\n\n";
    return (cost<thresh);
}

double Tracker::recencyCalc(double featureDet)
{
    //return exp(1-(recency/totalDet))/1.718 - 0.582;
    //return (1-recency/totalDet);
    return 1.582 - exp(recency/featureDet)/1.718;
}

double Tracker::frequencyCalc(double totalDet)
{
    return log(num_detections)/log(totalDet);
}

double Tracker::getThreshold(){return mahalanobis_threshold+oversaturation_penalty;}