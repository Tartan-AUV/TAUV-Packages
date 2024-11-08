#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <tauv_msgs/FeatureDetection.h>
#include <tauv_msgs/FeatureDetections.h>
#include <std_srvs/Trigger.h>
#include <tauv_msgs/FeatureDetectionsSync.h>
#include <tauv_msgs/MapFind.h>
#include <tauv_msgs/MapFindClosest.h>
#include <tauv_msgs/MapFindOne.h>
#include <unordered_map>
#include <string>
#include <utility>
#include <iostream>
#include <mutex>
#include <queue>
#include <memory>
#include "kalman-filter/kalman_filter.hpp"

#define UNKNOWN_KEYWORD "unknown"

/**
 * Mapping system for any feature detections. Uses Kalman filtering and similarity matching to derive better estimates
 * of objects over time. Uses detection count and recency decay to filter sparse false detections.
 * All tunable parameters are in global_map.yaml
**/

using namespace std;

class TrackerMaster;
class Tracker;

/**
 * An internal C++ definition of detections to avoid reinitializing new Eigen vectors instead of reusing the 
 * same struct internally to do math and working with mathematically inconvenient tauv_msgs only for external purposes
**/
struct Detection {
    Eigen::Vector3d position;
    Eigen::Vector3d orientation;
    /**
     * tag = UNKNOWN_KEYWORD is a keyword and will be considered for matches with any tracker
     * */
    string tag;
    /**
     * Used for tuning Tracker settings (see global_map.yaml params) if a new tracker for the detection is formed
    **/
    string tracker_type;
    /**
     * Detection position confidence [0.0,1.0]. Used in adjusting the rate of tracker position estimates based on new detections.
     * A confidence of 0.0 will be essentially overwritten by any nonzero confidence detection
    **/
    double confidence;

    //detection position is two_dimensional (uses SE2 simlarity comparisons, Kalman filter only tracks SE2 dimensions)
    bool SE2;
};

//struct conversion helper functions
Eigen::Vector3d point_to_vec(geometry_msgs::Point point);
geometry_msgs::Point vec_to_point(Eigen::Vector3d vec);

/**
 * Used to represent Trackers that have been retired from taking any additional input because of a lack of recency
 * and frequency of detections, which when low enough are assumed to mean the Tracker's detections were false positives
 * A Tracker can only be retired after it becomes a potential zombie, which happens when no new matches have been made 
 * after that tracker's expected matching period (which should correspond to min_decay_time).
 **/
enum TrackerState {
    ACTIVE = 0,
    ZOMBIE = 1, //retired
    POTENTIAL_ZOMBIE = 2 //tracker has gone over min decay time and no longer is obligated to be present 
};

/**
 * A Tracker represents a single detected object that was measured with error over time. This class performs the
 * feature-specific Kalman filtering and is the "bucket", where all the different feature measurements are sent
 * and combined for a better estimate of the object's position and orientation.
 * A single instance tracks a single, identified map feature and its matched detections
 **/
class Tracker : public enable_shared_from_this<Tracker>
{
    public:
        Tracker(Detection &initial_detection, ros::NodeHandle& handler);
        Tracker(Detection &initial_detection, ros::NodeHandle& handler, shared_ptr<KalmanFilter> position, shared_ptr<KalmanFilter> orientation);
        
        shared_ptr<Tracker> makeReassignment(Detection &detection);
        bool reassignable(Detection &detection);

        //convenience functions, coordinates need retrieval from Kalman filter
        Eigen::Vector3d getPosition();
        Eigen::Vector3d getOrientation();
        double getConfidence();
        size_t getNumDetections();
        
        bool validCost(double cost, bool oversaturated);

        void addDetection(Detection &detection);
        double getSimilarityCost(Detection &det, size_t total_num_detections);
        double getThreshold();

        void incrementRecency();
        void setNumDetections(int num);
        string getTag();
        string getTrackerType();
        bool is_SE2();

        double frequencyCalc(double totalDet);
        double recencyCalc(double totalDet);

        /**
         * reset is used to put a tracker into "ZOMBIE" mode (wipes data and ready to be reinitialized with new detections)
         * reinit is used to revive the tracker with completely new data
         * same as deleting and making a new tracker but slightly more efficient with regards to TrackerMaster that owns the Tracker
        **/
        void reset();
        void reinit(Detection &initial_detection);
        void setPotentialZombie(const ros::TimerEvent& event);

        TrackerState State;

    private:
        void initialize(Detection &initial_detection, ros::NodeHandle& handler);
        double getParam(string property, double def=0);
        bool getParam(string property, bool def);
        void readParams();
        void resetTimer();

        double getDistance(Detection &det);
        double getRotation(Detection& detection);
        bool diffTag(string tag);
        bool diffTracker(string tracker_tag);

        /**
        * more computationally efficient to separate position and orientation predictions
        * for the matrix inversion required in Kalman filtering
        **/
        shared_ptr<KalmanFilter> kPosition;
        shared_ptr<KalmanFilter> kOrientation;

        size_t num_detections;
        double recency;

        string feature_tag;
        string tracker_type;
        bool reassignable_tracker;
        bool SE2;

        //decay weight
        double min_decay_time;

        //matching weights
        double mahalanobis_threshold;
        double tag_weight;
        double distance_weight;
        double orientation_weight;
        double recency_matching_weight;
        double frequency_matching_weight;
        double confidence_weight;
        double oversaturation_penalty;
        double tracker_bias;

        ros::Timer decayTimer; //tracks elapsed time between detections and creates a potential zombie if overtime
        ros::NodeHandle nodeHandler;
};


/**
 * One TrackerMaster owns all Trackers for a particular tag.
 * This parent class specializes in setting up matching and retiring unused Trackers.
 * It generates the similarity matching costs for each of its Tracker slaves. 
 * By controlling all Trackers of a tag from one tag master, we can perform tag-specific matching protocols by allowing 
 * different tags to have different matching weights (and thus allow different tags to prioritize matching in different ways).
 * We can also use this class to make sure we aren't generating more Trackers than we expect for a particular tag 
 * (if we know we have two badges, we should be creating a penalty or simply not allowing more than two badges).
 * This class serves as the interface between the GlobalMap and Trackers, which are focused only on filtering and
 * processing information from any detections they are given for the single object they are modelling.
**/
class TrackerMaster : public enable_shared_from_this<TrackerMaster>
{
    public:
        TrackerMaster(string tag, ros::NodeHandle& handler);
        shared_ptr<Tracker> getTracker(size_t featureIdx);

        vector<shared_ptr<Tracker>> getTrackers();
        size_t getNumTrackers();
        double getDecay(size_t featureIdx, int totalDetections);
        double getMaxThreshold();

        //for interfacing with Trackers
        void addDetection(size_t featureIdx, Detection &detection);

        void addTracker(Detection &detection);
        void addTracker(shared_ptr<Tracker> F);

        //for reassignment of Trackers
        shared_ptr<Tracker> popReassignable(size_t featureIdx, Detection& det);
        bool reassignable(size_t featureIdx, Detection &detection);

        /**For Zombifying Trackers when their decay (some tunable combination of frequency and recency) score is too low.
        * Creating Zombies allows us to avoid interfering with the Tracker List while the Global Map is commanding matching,
        * thus preserving safe memory accesses, and creates a way to reuse already setup Trackers. 
        * Global Map commands updates to Tracker List only prior to assignment and matching.
        **/
        void deleteTracker(size_t featureIdx);
        double decay(size_t featureIdx, int totalDetections);

        //for deleting Trackers when there are too many Zombies
        void makeUpdates();

        //matching information for Global Map
        bool validCost(size_t featureIdx, double cost);
        int generateSimilarityMatrix(vector<Detection> &detections, vector<vector<double>> &costMatrix, size_t trackerNum, vector<pair<shared_ptr<TrackerMaster>, int>> &trackerList);

    private:
        double getParam(string property, double def = 0.0);
        vector<double> getSimilarityRow(shared_ptr<Tracker> F, vector<Detection> &detections);

        double max_threshold;

        double DECAY_THRESHOLD;
        double recency_weight;
        double frequency_weight;

        int predictedTrackerNum;
        int numDetections;

        /**All features under this class have this same feature_tag tag.
        * BUT, there is leniency as dictated by the given ROS params in matching, where
        * detections that have different tags may be matched with trackers of a particular tag.
        * This is deliberate to allow user flexibility in determining how tags can be matched 
        * (e.g., to allow for imperfect labelling.)
        * */
        string feature_tag;

        vector<shared_ptr<Tracker>> TrackerList;
        priority_queue<size_t, vector<size_t>, std::greater<size_t>> Zombies;

        //new zombies in queue!!!
        bool needsUpdating;

        ros::NodeHandle nodeHandler;
};


/**
 * GlobalMap communicates with rest of system and manages high-level matching of detections to trackers.
 * It is an interface for Tracker search and commands the process flow of detection distribution.
 * Essentially tells TrackerMasters who to assign detections to (its a whole bureaucracy) after getting
 * similarity information from them.
**/
class GlobalMap
{
    public:
        GlobalMap(ros::NodeHandle& handler);

        void updateTrackersInterface(const tauv_msgs::FeatureDetections::ConstPtr& detectionObjects);
        bool updateTrackers(vector<tauv_msgs::FeatureDetection> objdets, string detector_type);
        void addTracker(Detection &detection);

        //service callback functions
        //returns all detections matching selected tag
        bool find(tauv_msgs::MapFind::Request &req, tauv_msgs::MapFind::Response &res);
        //returns the most recent/highest count object with a certain tag
        bool findOne(tauv_msgs::MapFindOne::Request &req, tauv_msgs::MapFindOne::Response &res);
        //returns the tracker closest to the passed point estimate with a matching tag
        bool findClosest(tauv_msgs::MapFindClosest::Request &req, tauv_msgs::MapFindClosest::Response &res);
        bool reset(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);

        //for synchronous interactions w map
        bool syncAddDetections(tauv_msgs::FeatureDetectionsSync::Request &req, tauv_msgs::FeatureDetectionsSync::Response &res); 

    private:
        shared_ptr<TrackerMaster> findTrackerMaster(string tag);
        void assignDetection(shared_ptr<TrackerMaster> F, size_t featureIdx, Detection &detection);

        void assignDetections(vector<Detection> &detections);
        vector<Detection> convertToStruct(vector<tauv_msgs::FeatureDetection> &detections, string tracker_type);
        vector<pair<shared_ptr<TrackerMaster>, int>> generateSimilarityMatrix(vector<Detection> &detections, vector<vector<double>> &costMatrix, size_t costMatSize);

        unordered_map<string, shared_ptr<TrackerMaster>> MAP;
        void updateDecay(shared_ptr<TrackerMaster> F, size_t featureIdx);

        ros::Subscriber listener;
        ros::ServiceServer resetService;
        ros::ServiceServer syncDetectionsService;
        ros::ServiceServer findService;
        ros::ServiceServer findOneService;
        ros::ServiceServer findClosestService;
        ros::NodeHandle nodeHandler;
        mutex mtx;

        size_t featureCount;
        size_t totalDetections;
        double  DUMMY_FILL;
};