#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <limits.h>
#include <assert.h>
#include <memory.h>
#include <getopt.h>
#include <limits.h>
#include <pthread.h>
#include <sys/sysinfo.h>
#include <string>
#include <errno.h>



#ifndef ASSERT
#define ASSERT assert
#endif
#define USECS_PER_SECOND	1000000UL


#ifndef _countof
#define _countof(x) (sizeof(x) / sizeof(x[0]))
#endif

#ifndef BDIO_STRINGIZE1
#       define BDIO_STRINGIZE1(x) #x
#endif
#ifndef BDIO_STRINGIZE
#       define BDIO_STRINGIZE(x) BDIO_STRINGIZE1(x)
#endif
#define BDIO_API
#define BD_sscanf sscanf
typedef int BDError ;
static const BDError BD_SUCCESS = 0 ;


BDError BD_GetError(void)
{
	return errno ;
}

/**
 *  Enumerated Type for thread priority
 */
enum ThreadPriority 
{
	PRIORITY_UNKNOWN = -1,  /**< Unspecified  */
	PRIORITY_MAX,      /**< The maximum possible priority  */
	PRIORITY_HIGH,     /**< A high (but not max) setting   */
	PRIORITY_NOMINAL,  /**< An average priority            */
	PRIORITY_LOW,      /**< A low (but not min) setting    */
	PRIORITY_MIN,      /**< The miniumum possible priority */
	PRIORITY_DEFAULT   /**< Priority scheduling default    */
};

#define PRIORITY_UNKNOWN_STR      BDIO_STRINGIZE(PRIORITY_UNKNOWN)
#define PRIORITY_MAX_STR          BDIO_STRINGIZE(PRIORITY_MAX)
#define PRIORITY_HIGH_STR         BDIO_STRINGIZE(PRIORITY_HIGH)
#define PRIORITY_NOMINAL_STR      BDIO_STRINGIZE(PRIORITY_NOMINAL)
#define PRIORITY_LOW_STR          BDIO_STRINGIZE(PRIORITY_LOW)
#define PRIORITY_MIN_STR          BDIO_STRINGIZE(PRIORITY_MIN)
#define PRIORITY_DEFAULT_STR      BDIO_STRINGIZE(PRIORITY_DEFAULT)

/**
 *  Enumerated Type for thread scheduling policy
 */
enum ThreadPolicy 
{
	SCHEDULE_UNKNOWN = -1,     /**< Unspecified                            */
	SCHEDULE_TIME_SHARE,  /**< Time-share scheduling (IRIX DEFAULT)   */
	SCHEDULE_FIFO,        /**< First in, First out scheduling         */
	SCHEDULE_ROUND_ROBIN, /**< Round-robin scheduling (LINUX_DEFAULT) */ 
	SCHEDULE_DEFAULT      /**< Default scheduling                     */
};

#define SCHEDULE_UNKNOWN_STR          BDIO_STRINGIZE(SCHEDULE_UNKNOWN)
#define SCHEDULE_FIFO_STR             BDIO_STRINGIZE(SCHEDULE_FIFO)
#define SCHEDULE_ROUND_ROBIN_STR      BDIO_STRINGIZE(SCHEDULE_ROUND_ROBIN)
#define SCHEDULE_TIME_SHARE_STR       BDIO_STRINGIZE(SCHEDULE_TIME_SHARE)
#define SCHEDULE_DEFAULT_STR          BDIO_STRINGIZE(SCHEDULE_DEFAULT)

/**
 *  Set the current thread's schedule priority.  This is a complex method.
 *  Beware of thread priorities when using a many-to-many kernel
 *  entity implemenation (such as IRIX pthreads).  If one is not carefull
 *  to manage the thread priorities, a priority inversion deadlock can
 *  easily occur Unless you have explicit need to set the schedule 
 *  priorites for a given task, it is best to leave them alone.
 *
 *  @note some implementations (notably LinuxThreads and IRIX Sprocs) 
 *  only alow you to decrease thread priorities dynamically.  Thus,
 *  a lower priority thread will not allow it's priority to be raised
 *  on the fly.  
 *
 *  @return BD_SUCCESS if normal.
 */
BDIO_API BDError SetCurrentSchedulePriority(ThreadPriority priority);
BDIO_API BDError SetCurrentSchedulePriority(const std::string& priority);

BDIO_API ThreadPriority ThreadPriorityFromString(const std::string& priority) ;
BDIO_API std::string ThreadPriorityToString(ThreadPriority priority) ;


/**
 *  Get the thread's schedule priority (if able)
 *
 *
 *  @return thread priority if normal, PRIORITY_UNKNOWN if error and BD_GetError set.
 */
BDIO_API ThreadPriority GetCurrentSchedulePriority();

/**
 *  Set the thread's scheduling policy (if able)
 *  
 *  @note On some implementations (notably IRIX Sprocs & LinuxThreads) 
 *  The policy may prohibit the use of SCHEDULE_ROUND_ROBIN and
 *  SCHEDULE_FIFO policies - due to their real-time nature, and 
 *  the danger of deadlocking the machine when used as super-user.  
 *  In such cases, the command is a no-op.
 *
 *  @return BD_SUCCESS if normal.
 */
BDIO_API BDError SetCurrentSchedulePolicy(ThreadPolicy policy);
BDIO_API BDError SetCurrentSchedulePolicy(const std::string& policy);

BDIO_API ThreadPolicy ThreadPolicyFromString(const std::string& policy) ;
BDIO_API std::string ThreadPolicyToString(ThreadPolicy policy) ;



/**
 *  Get the thread's policy (if able)
 *
 *  @return policy if normal, SCHEDULE_UNKNOWN if error and BD_GetError set.
 */
BDIO_API ThreadPolicy GetCurrentSchedulePolicy();

struct ThreadPriorityEntry
{
	const char * name ;
	ThreadPriority priority ;
} ;

ThreadPriorityEntry threadPriorityTable[] = {
	{PRIORITY_UNKNOWN_STR,    PRIORITY_UNKNOWN},
	{PRIORITY_MAX_STR,        PRIORITY_MAX},
	{PRIORITY_HIGH_STR,       PRIORITY_HIGH},
	{PRIORITY_NOMINAL_STR,    PRIORITY_NOMINAL},
	{PRIORITY_LOW_STR,        PRIORITY_LOW},
	{PRIORITY_MIN_STR,        PRIORITY_MIN},
	{PRIORITY_DEFAULT_STR,    PRIORITY_DEFAULT}
} ;

struct ThreadPolictEntry
{
	const char * name ;
	ThreadPolicy policy ;
} ;

ThreadPolictEntry threadPolicyTable[] = {
	{SCHEDULE_UNKNOWN_STR,      SCHEDULE_UNKNOWN},
	{SCHEDULE_FIFO_STR,         SCHEDULE_FIFO},
	{SCHEDULE_ROUND_ROBIN_STR,  SCHEDULE_ROUND_ROBIN},
	{SCHEDULE_TIME_SHARE_STR,   SCHEDULE_TIME_SHARE},
	{SCHEDULE_DEFAULT_STR,      SCHEDULE_DEFAULT}
} ;

ThreadPriority ThreadPriorityFromString(const std::string& priority)
{
	ThreadPriority answer = PRIORITY_UNKNOWN ;
	int th_priority = 0 ;
	ssize_t rv = BD_sscanf(priority.c_str(), "%d", &th_priority) ;
	if(rv == 1)
	{
		answer = ThreadPriority(th_priority) ;
	} else
	{
		for(size_t i=0; i < _countof(threadPriorityTable); i++)
		{
			if(threadPriorityTable[i].name == priority)
			{
				answer = threadPriorityTable[i].priority ;
				break ;
			}
		}
	}
	return answer ;
}
std::string ThreadPriorityToString(ThreadPriority priority)
{
	std::string answer = PRIORITY_UNKNOWN_STR ;
	for(size_t i=0; i < _countof(threadPriorityTable); i++)
	{
		if(threadPriorityTable[i].priority == priority)
		{
			answer = threadPriorityTable[i].name ;
			break ;
		}
	}
	return answer ;
	
}

ThreadPolicy ThreadPolicyFromString(const std::string& policy)
{
	ThreadPolicy answer = SCHEDULE_UNKNOWN ;
	int th_policy = 0 ;
	ssize_t rv = BD_sscanf(policy.c_str(), "%d", &th_policy) ;
	if(rv == 1)
	{
		answer = ThreadPolicy(th_policy) ;
	} else
	{
		for(size_t i=0; i < _countof(threadPolicyTable); i++)
		{
			if(threadPolicyTable[i].name == policy)
			{
				answer = threadPolicyTable[i].policy ;
				break ;
			}
		}
	}
	return answer ;
}
std::string ThreadPolicyToString(ThreadPolicy policy)
{
	std::string answer = SCHEDULE_UNKNOWN_STR ;
	for(size_t i=0; i < _countof(threadPolicyTable); i++)
	{
		if(threadPolicyTable[i].policy == policy)
		{
			answer = threadPolicyTable[i].name ;
			break ;
		}
	}
	return answer ;
}

BDError SetCurrentSchedulePriority(const std::string& priority)
{
	BDError answer = BD_SUCCESS ;
	int th_priority = 0 ;
	ssize_t rv = BD_sscanf(priority.c_str(), "%d", &th_priority) ;
	if(rv == 1)
	{
		int th_policy = 0 ;
		sched_param th_param;
		pthread_getschedparam(pthread_self(), &th_policy, &th_param);
		int max_priority = sched_get_priority_max(th_policy);
		int min_priority = sched_get_priority_min(th_policy);
		if(th_priority >= min_priority && th_priority <= max_priority)
		{
			th_param.sched_priority = th_priority;
			int result = pthread_setschedparam(pthread_self(), th_policy, &th_param);
			if(result != 0)
			{
				answer = BD_GetError() ;
			}
		}
	} else
	{
		answer = SetCurrentSchedulePriority(ThreadPriorityFromString(priority)) ;
	}
	return answer ;
}
BDError SetCurrentSchedulePolicy(const std::string& policy)
{
	return SetCurrentSchedulePolicy(ThreadPolicyFromString(policy)) ;
}

BDError SetCurrentSchedulePriority(ThreadPriority priority)
{
	BDError answer = BD_SUCCESS ;
	if(priority != PRIORITY_DEFAULT /* && priority != PRIORITY_UNKNOWN */)
	{
		int th_policy = 0 ;
		sched_param th_param;
		pthread_getschedparam(pthread_self(), &th_policy, &th_param);

		int max_priority = sched_get_priority_max(th_policy);
		int min_priority = sched_get_priority_min(th_policy);
		int nominal_priority = (max_priority + min_priority)/2;

		switch(priority) {
		
			case PRIORITY_MAX:
				th_param.sched_priority = max_priority;
				break;
			
			case PRIORITY_HIGH:
				th_param.sched_priority = (max_priority + nominal_priority)/2;
				break;
			
			case PRIORITY_NOMINAL:
				th_param.sched_priority = nominal_priority;
				break;   
			
			case PRIORITY_LOW:
				th_param.sched_priority = (min_priority + nominal_priority)/2;
				break;       
			
			case PRIORITY_MIN:
				th_param.sched_priority = min_priority;
				break;   
			
			case PRIORITY_DEFAULT:
				th_param.sched_priority = nominal_priority;
				break;   
			
			default:
				th_param.sched_priority = min_priority;
				break;  
		
		}
		 
		int rv = pthread_setschedparam(pthread_self(), th_policy, &th_param);
		if(rv != 0)
		{
			answer = BD_GetError() ;
		}
	}

	return answer ;
}
ThreadPriority GetCurrentSchedulePriority()
{
	ThreadPriority answer = PRIORITY_UNKNOWN ;

	return answer ;
}
BDError SetCurrentSchedulePolicy(ThreadPolicy policy)
{
	BDError answer = BD_SUCCESS ;

	if(policy != SCHEDULE_DEFAULT /* && policy != SCHEDULE_UNKNOWN */)
	{
		int th_policy = SCHED_OTHER ;
		sched_param th_param;
		pthread_getschedparam(pthread_self(), &th_policy, &th_param);
		int max_priority = sched_get_priority_max(policy);
		int min_priority = sched_get_priority_min(policy);
		int nominal_priority = (max_priority + min_priority)/2;

		switch(policy) 
		{
			case SCHEDULE_FIFO:
				th_policy = SCHED_FIFO;
				th_param.sched_priority = nominal_priority;
				break;

			case SCHEDULE_ROUND_ROBIN:
				th_policy = SCHED_RR;
				th_param.sched_priority = nominal_priority;
				break;

			case SCHEDULE_TIME_SHARE:
				th_policy = SCHED_OTHER;
				th_param.sched_priority = 0;
				break;

			case SCHEDULE_DEFAULT:
				th_policy = SCHED_OTHER;
				th_param.sched_priority = 0;
				break;

			default:
#ifdef __sgi
				th_policy = SCHED_RR;
#else
				th_policy = SCHED_OTHER;
				th_param.sched_priority = 0;
#endif
				break;
		};

		int rv = pthread_setschedparam(pthread_self(), th_policy, &th_param);
		if(rv != 0)
		{
			answer = BD_GetError() ;
		}
	}
	return answer ;
}
ThreadPolicy GetCurrentSchedulePolicy()
{
	ThreadPolicy answer = SCHEDULE_UNKNOWN ;
	return answer ;
}


/*
 * CPUID instruction 0xb ebx info.
 */
#define	CPUID_TYPE_INVAL	0
#define	CPUID_TYPE_SMT		1
#define	CPUID_TYPE_CORE		2

static __inline void
cpuid_count(u_int ax, u_int cx, u_int *p)
{
    __asm __volatile("cpuid"
                     : "=a" (p[0]), "=b" (p[1]), "=c" (p[2]), "=d" (p[3])
                     :  "0" (ax), "c" (cx));
}


typedef struct sk_atomics_pair {
    uint32_t	   counter32;
    uint64_t	   counter64;
} sk_atomics_pair_t;

typedef union sk_atomics_pair_padded_u {
    sk_atomics_pair_t	   locks;
    char pad[4096];
} sk_atomics_pair_padded_t;

sk_atomics_pair_padded_t lock_buffers[2];
sk_atomics_pair_padded_t *lock_array[2];

typedef enum perf_test_type {
	counter32_test,
	counter64_test,
	memory_stride0,
	memory_set_odds,
	memory_set_evens,
	memory_set_mix,
} perf_test_type_t;

static const char * perf_test_type_str(perf_test_type_t test_type) {
	switch(test_type) {
	case counter32_test:
		return "counter32_test";
	case counter64_test:
		return "counter64_test";
	case memory_stride0:
		return "memory_stride0";
	case memory_set_odds:
		return "memory_set_odds";
	case memory_set_evens:
		return "memory_set_evens";
	case memory_set_mix:
		return "memory_set_mix";
	default:
		return "Unknown";
		break;
	}
}

struct sk_Msg {
    uint16_t sk_flags;		/* exception state && flags */
    union {			/* Message data. */
        void *p;
        int l;
        uint64_t ll;
    } data;
};

#define _THREAD_NAME_SIZE       128

typedef struct _thread_msg{
    struct _thread_msg *next;
    void *data;
} _thread_msg_t;

typedef struct _thread_info{
    size_t thread_num;
    size_t secs_to_run;
    _thread_info *parent_pid;

    pthread_t t_id;
    pthread_attr_t t_attr;
    char t_name[_THREAD_NAME_SIZE];
    pthread_mutex_t msg_mutex;
    pthread_cond_t msg_cv;
    _thread_msg_t *msg_head;
    _thread_msg_t *msg_tail;

    pthread_mutex_t ready_lock;
    pthread_cond_t ready_cv;
    uint64_t start_cycles;
    uint64_t end_cycles;
    uint64_t ops;
    sk_atomics_pair_padded_t *lock;
    perf_test_type_t test_type;
    int x2acpi_id;
    unsigned SMTSelectMask;
    unsigned SMTMaskWidth;
    unsigned PkgSelectMask;
    unsigned PkgSelectMaskShift;
    unsigned CoreSelectMask;
    unsigned pkg_IDAPIC;
    unsigned Core_IDAPIC;
    unsigned SMT_IDAPIC;
} thread_info_t;

thread_info_t thread_infos[__CPU_SETSIZE];

static inline long lock_test_npcpu(void) {
    return get_nprocs_conf();
}

static inline long lock_test_nvcpu(void) {
    return get_nprocs();
}

static uint64_t
lock_test_get_available_memsize(void)
{
	return (uint64_t)sysconf(_SC_PHYS_PAGES) * (uint64_t)sysconf(_SC_PAGESIZE);
}

#define tsc_freq (sk_cycles_per_msec * 1000UL)

typedef struct {
    uint64_t total_cycles;
    uint64_t total_ops;
} thread_info_stats_t;

uint64_t seed = 0xFadedFacade;

static uint64_t rng64(uint64_t *s)
{
    uint64_t c = 7319936632422683419ULL;
    uint64_t x = s[1];

	union u128_64
	{
	    uint64_t seed[2];
	    __uint128_t val;
	};

	/* Increment 128bit counter */
	((union u128_64 *)s)->val += c + ((__uint128_t) c << 64);

	/* Two h iterations */
	x ^= (x >> 32) ^ (uint64_t) s;
	x *= c;
	x ^= x >> 32;
	x *= c;

	/* Perturb result */
	return x + s[0];
}

static uint64_t rand64(void) {
	uint64_t r = rng64(&seed);
	return r;
}

static unsigned num_packages = 0;
static unsigned num_cores_per_package = 0;

static uint64_t memory_size;
static uint64_t buffer_size;
static uint64_t num_pages;
static uint8_t * test_buffer = NULL;
static pthread_mutexattr_t _mutex_attr;

static void* lock_test_thread(void *);

static uint64_t sk_cycles_per_msec = 0;
static uint64_t sk_cycles_per_usec = 0;

/* returns a buffer of struct timespec with the time difference of start and stop
   stop is assumed to be greater than start */
static void
timespec_diff(const struct timespec *start, const struct timespec *stop,
                   struct timespec *result)
{
    if ((stop->tv_nsec - start->tv_nsec) < 0) {
        result->tv_sec = stop->tv_sec - start->tv_sec - 1;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
    } else {
        result->tv_sec = stop->tv_sec - start->tv_sec;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec;
    }

    return;
}
static inline uint64_t sk_cycles(void)
{
	unsigned int lo, hi;
	__asm__ volatile("rdtsc" : "=a" (lo), "=d" (hi));
	return lo | ((uint64_t)hi << 32);
}

struct cyccounter_timer_t
{
    uint64_t start, stop;
};
typedef struct cyccounter_timer_t cyccounter_timer_t;

static void sk_init_cycles(void)
{
	cyccounter_timer_t timer;
	struct timespec begints, endts;
	uint64_t nsecElapsed;
	uint64_t gettime_cost;
	struct timespec tmpts;

	sleep(0);

	timer.start = sk_cycles();
	timer.stop = sk_cycles();
	uint64_t cyccounter_cost = timer.stop - timer.start;

	clock_gettime(CLOCK_MONOTONIC, &begints);
	clock_gettime(CLOCK_MONOTONIC, &endts);
	timespec_diff(&begints, &endts, &tmpts);
	gettime_cost = tmpts.tv_sec * 1000000000LL + tmpts.tv_nsec;

	clock_gettime(CLOCK_MONOTONIC, &begints);
	timer.start = sk_cycles();
	usleep(1000 * 50) ;
	clock_gettime(CLOCK_MONOTONIC, &endts);
	timer.stop = sk_cycles();

	timespec_diff(&begints, &endts, &tmpts);
	nsecElapsed = tmpts.tv_sec * 1000000000LL + tmpts.tv_nsec;

	sk_cycles_per_usec = ((timer.stop - timer.start) - cyccounter_cost) * 1000LL / (nsecElapsed - gettime_cost);
	sk_cycles_per_msec = ((timer.stop - timer.start) - cyccounter_cost) * 1000000LL / (nsecElapsed - gettime_cost);
}

thread_info_t main_thread;

static void init_thread_msg_info(thread_info_t * thread_info) {
	thread_info->msg_head = NULL;
	thread_info->msg_tail = NULL;
    int ret = pthread_mutex_init(&thread_info->msg_mutex, NULL);
    ASSERT(ret == 0);
    ret = pthread_cond_init(&thread_info->msg_cv, NULL);
    ASSERT(ret == 0);
}

static
void setup_test(void) {
	int i=0;
	if(num_packages == 0) {
		if(((((uintptr_t)&lock_buffers[0]) >> 12) & 0) == 0) {
			lock_array[0] = &lock_buffers[0];
			lock_array[1] = &lock_buffers[1];
		} else {
			lock_array[0] = &lock_buffers[1];
			lock_array[1] = &lock_buffers[0];
		}
		memory_size = lock_test_get_available_memsize();
		buffer_size = memory_size / 16;
		if(buffer_size > 256 * 1024 * 1024) {
			buffer_size = 256 * 1024 * 1024;
		}
		num_pages = buffer_size / 4096;
		buffer_size = num_pages * 4096;

	    pthread_mutexattr_init(&_mutex_attr);
	    /* don't turn this on, we get CV failures */
	    pthread_mutexattr_settype(&_mutex_attr, PTHREAD_MUTEX_ERRORCHECK);
	    pthread_mutexattr_settype(&_mutex_attr, PTHREAD_MUTEX_NORMAL);
	    int ret = 0;
		init_thread_msg_info(&main_thread);

		for(i=0; i < lock_test_nvcpu(); i++) {
			if(!thread_infos[i].t_id) {
				thread_infos[i].thread_num = i;
				thread_infos[i].parent_pid = &main_thread;
				thread_infos[i].secs_to_run = 0;
			    pthread_mutex_init(&thread_infos[i].ready_lock, &_mutex_attr);
			    pthread_cond_init(&thread_infos[i].ready_cv, NULL);
				snprintf(thread_infos[i].t_name, sizeof(thread_infos[i].t_name), "lock_test_thread_%u", i);
				init_thread_msg_info(&thread_infos[i]);
			    ret  = pthread_attr_init(&thread_infos[i].t_attr);
			    ret |= pthread_create(&thread_infos[i].t_id,
			                          &thread_infos[i].t_attr,
									  lock_test_thread,
			                          (void*)(uint64_t)i);
			    ret |= pthread_detach(thread_infos[i].t_id);
			    ASSERT(ret == 0);

//			    thread_infos[i].pid = sk_create_bsd(lock_test_thread, thread_infos[i].thread_name,
//						SK_ALLOC_VSTACK, 32*1024, i, 0);
			    ret = pthread_mutex_lock(&thread_infos[i].ready_lock);
			    //Can not use pthread owner since b/c of cond_wait
			    ASSERT(ret == 0);
			    ret = pthread_cond_wait(&thread_infos[i].ready_cv, &thread_infos[i].ready_lock);
			    ASSERT(ret == 0);
			}
			ASSERT(thread_infos[i].t_id);
		}

		unsigned cur_package = UINT_MAX;
		i = 0;
		printf("        cpu    acpi_id        pkg       core        htt   smt_mask   smt_bits   pkg_mask  pkg_shift  core_mask\n");
		while(i < lock_test_nvcpu()) {
			if(thread_infos[i].pkg_IDAPIC == cur_package) {
				num_cores_per_package++;
				printf("         %2u         %2x         %2u         %2u         %2u       0x%02x       0x%02x 0x%02x       0x%02x       0x%02x\n",
						i,
						thread_infos[i].x2acpi_id,
						thread_infos[i].pkg_IDAPIC,
						thread_infos[i].Core_IDAPIC,
						thread_infos[i].SMT_IDAPIC,
						thread_infos[i].SMTSelectMask,
						thread_infos[i].SMTMaskWidth,
						thread_infos[i].PkgSelectMask,
						thread_infos[i].PkgSelectMaskShift,
						thread_infos[i].CoreSelectMask);
				i++;
			} else {
				cur_package = thread_infos[i].pkg_IDAPIC;
				num_cores_per_package = 0;
				num_packages++;
			}
		}
		printf("packages: %2u, cores_per_package: %2u\n", num_packages, num_cores_per_package);
	}

	for(i=0; i < (long)_countof(lock_array); i++) {
		lock_array[i]->locks.counter32 = 0;
		lock_array[i]->locks.counter64 = 0;
	}
	for(i=0; i < lock_test_nvcpu(); i++) {
		thread_infos[i].end_cycles = thread_infos[i].start_cycles = thread_infos[i].ops = 0;
	}

}

static void *
sk_receive_msg(thread_info_t* thread_info)
{
	_thread_msg_t *tm;
	void *msg;
	int err;

	pthread_mutex_lock(&thread_info->msg_mutex);
	again:
	if (thread_info->msg_head == NULL) {
		err = pthread_cond_wait(&thread_info->msg_cv, &thread_info->msg_mutex);
		ASSERT(err == 0);
		goto again;
	} else {
		/* there are pending msgs */
		tm = thread_info->msg_head;
		thread_info->msg_head = tm->next;
		if (tm->next == NULL) {
			thread_info->msg_tail = NULL;
		}
	}
	pthread_mutex_unlock(&thread_info->msg_mutex);

	msg = tm->data;
	free(tm);
	ASSERT(msg);

	return msg;
}

static void
sk_send_msg(thread_info_t* thread_info, void *msg)
{
	_thread_msg_t *tm;
	int err;

	tm = (_thread_msg_t *)malloc(sizeof(_thread_msg_t));

	ASSERT(tm);

	tm->data = msg;
	tm->next = NULL;

	pthread_mutex_lock(&thread_info->msg_mutex);
	if (thread_info->msg_tail == NULL) {
		thread_info->msg_head = tm;
		thread_info->msg_tail = tm;
		err = pthread_cond_broadcast(&thread_info->msg_cv);
		ASSERT(err == 0);
	} else {
		/* there are pending msgs */
		thread_info->msg_tail->next = tm;
		thread_info->msg_tail = tm;
	}
	pthread_mutex_unlock(&thread_info->msg_mutex);

}

static void* lock_test_thread(void * arg) {
	uint64_t cpuid = (uint64_t)arg;
	thread_info_t * my_thread_info = &thread_infos[cpuid];
	int ret = 0;

    cpu_set_t currentCPU;
    CPU_ZERO(&currentCPU);

    /* turn on the equivalent bit inside the bitmap corresponding to affinitymask */
    CPU_SET(cpuid, &currentCPU);

    pthread_setaffinity_np (my_thread_info->t_id, sizeof(currentCPU), &currentCPU);

//    sk_bind_processor_pid(sk_my_pid(), (int)cpuid % lock_test_npcpu());
    sched_yield();
	ASSERT(sched_getcpu() == (cpuid % lock_test_npcpu()));
	ret = SetCurrentSchedulePolicy(SCHEDULE_FIFO);
	if(ret != 0) {
		perror("SetCurrentSchedulePolicy");
	}
	ASSERT(ret == 0);
	ret = SetCurrentSchedulePriority(PRIORITY_MAX);
	if(ret != 0) {
		perror("SetCurrentSchedulePriority");
	}
	ASSERT(ret == 0);
#if defined(__APPLE__)
    ret = pthread_setname_np(my_thread_info->t_name);
#elif defined(__linux__)
    ret = pthread_setname_np(pthread_self(), my_thread_info->t_name);
#endif
    printf("%s running on thread %d\n", my_thread_info->t_name, sched_getcpu());

	unsigned p[4];
	cpuid_count(0, 0, p);
	unsigned maxCPUID = p[0];
	ASSERT(maxCPUID >= 0x0B);
	if(maxCPUID >= 0x0B) {
		int i;
		bool wasCoreReported = false;
		bool wasThreadReported = false;
		unsigned coreplusSMT_Mask = 0;
		/* We only support two levels for now. */
		for (i = 0; i < 3; i++) {
			cpuid_count(0x0B, i, p);
			if(p[1] == 0) {
				break;
			}
			thread_infos[cpuid].x2acpi_id = p[3];
			unsigned levelType = (p[2] >> 8) & 0xff;
			unsigned levelShift = p[0] & 0x1f;
			switch(levelType) {
			case CPUID_TYPE_SMT:
				thread_infos[cpuid].SMTSelectMask = ~(UINT_MAX << levelShift);
				thread_infos[cpuid].SMTMaskWidth = levelShift;
				wasThreadReported = true;
				break;
			case CPUID_TYPE_CORE:
				coreplusSMT_Mask = ~(UINT_MAX << levelShift);
				thread_infos[cpuid].PkgSelectMaskShift = levelShift;
				thread_infos[cpuid].PkgSelectMask = UINT_MAX ^ coreplusSMT_Mask;
				wasCoreReported = true;
				break;
			}
			if (wasThreadReported && wasCoreReported) {
				thread_infos[cpuid].CoreSelectMask = coreplusSMT_Mask ^ thread_infos[cpuid].SMTSelectMask;
			} else if (!wasCoreReported && wasThreadReported) {
				thread_infos[cpuid].CoreSelectMask = 0;
				thread_infos[cpuid].PkgSelectMaskShift =  thread_infos[cpuid].SMTMaskWidth;
				thread_infos[cpuid].PkgSelectMask = UINT_MAX ^ thread_infos[cpuid].SMTSelectMask;
			}
			thread_infos[cpuid].pkg_IDAPIC      = ((thread_infos[cpuid].x2acpi_id & thread_infos[cpuid].PkgSelectMask) >> thread_infos[cpuid].PkgSelectMaskShift);
			thread_infos[cpuid].Core_IDAPIC     = ((thread_infos[cpuid].x2acpi_id & thread_infos[cpuid].CoreSelectMask) >> thread_infos[cpuid].SMTMaskWidth);
			thread_infos[cpuid].SMT_IDAPIC      =  (thread_infos[cpuid].x2acpi_id & thread_infos[cpuid].SMTSelectMask);
		}
	}
#ifdef ONNIX
	thread_infos[cpuid].pkg_IDAPIC = (unsigned)cpuid / (lock_test_nvcpu() / 2);
	thread_infos[cpuid].Core_IDAPIC = (unsigned)cpuid % (lock_test_nvcpu() / 2);
#endif

    ret = pthread_mutex_lock(&my_thread_info->ready_lock);
    ASSERT(ret == 0);
    (void) pthread_cond_signal(&my_thread_info->ready_cv);
    ret = pthread_mutex_unlock(&my_thread_info->ready_lock);
    ASSERT(ret == 0);

//	sk_lock(&my_thread_info->ready_lock);
//	sk_cv_signal(&my_thread_info->ready_cv);
//	sk_unlock(&my_thread_info->ready_lock);

	for(;;) {
		//printf("%2lu.%12lu: Waiting for start msg\n", cpuid, get_cyclecount());
		sk_Msg *async_msg = (sk_Msg*)sk_receive_msg(my_thread_info);
		ASSERT(async_msg != NULL);
		ASSERT(sched_getcpu() == (cpuid % lock_test_npcpu()));
		sk_atomics_pair_padded_t *lock_pair = (sk_atomics_pair_padded_t *)async_msg->data.p;
		delete async_msg;
		async_msg = NULL;
		//printf("%2lu.%12lu: Beginning loop\n", cpuid, get_cyclecount());
		uint64_t cycles_to_run = my_thread_info->secs_to_run * sk_cycles_per_msec * 1000;
		//printf("%2lu.%12lu: Cycles to run: %lu\n", cpuid, get_cyclecount(), cycles_to_run);
		my_thread_info->start_cycles = sk_cycles();
		int min_loop_count = 2;
		int max_loop_count = 10;
		uint32_t page_id = rand64() % num_pages;
		while(sk_cycles() < my_thread_info->start_cycles + cycles_to_run) {
			switch(my_thread_info->test_type) {
			case counter32_test:
				if(lock_pair == NULL) {
					continue;
				}
				__sync_fetch_and_add(&lock_pair->locks.counter32, 1);
				break;
			case counter64_test:
				if(lock_pair == NULL) {
					continue;
				}
				__sync_fetch_and_add(&lock_pair->locks.counter32, 1);
				break;
			case memory_stride0: {
				page_id = page_id++ % num_pages;
				ASSERT(page_id < num_pages);
				uint8_t * counter_ptr = (test_buffer + (page_id * 4096));
				uint64_t * counter64_ptr = (uint64_t*)counter_ptr;
				__sync_fetch_and_add(counter64_ptr, 1);
				break;
			}
			case memory_set_odds: {
				uintptr_t test_buffer_page = (uintptr_t)test_buffer >> 12;
				page_id = (rand64() % (num_pages / 2)) * 2 + ((test_buffer_page & 1) ? 1 : 0);
				ASSERT(page_id < num_pages);
				uint8_t * buffer_ptr = (test_buffer + (page_id % num_pages) * 4096);
				memset(buffer_ptr, page_id, 512);
				break;
			}
			case memory_set_evens: {
				uintptr_t test_buffer_page = (uintptr_t)test_buffer >> 12;
				page_id = (rand64() % (num_pages / 2)) * 2 + ((test_buffer_page & 1) ? 0 : 1);
				ASSERT(page_id < num_pages);
				uint8_t * buffer_ptr = (test_buffer + (page_id % num_pages) * 4096);
				memset(buffer_ptr, page_id, 512);
				break;
			}
			case memory_set_mix: {
				page_id = rand64() % num_pages;
				ASSERT(page_id < num_pages);
				uint8_t * buffer_ptr = (test_buffer + (page_id % num_pages) * 4096);
				memset(buffer_ptr, page_id, 512);
				break;
			}
			}
			int c = (((max_loop_count - min_loop_count + 1) * rand()) / RAND_MAX) + min_loop_count;
			int n;
			uint64_t sum=0;
			for(n = 0; n < c; n++) {
				// sk_pause_execution();
				if(n < 2) {
					sum++;
				} else {
					sum += n;
				}
			}
			my_thread_info->ops++;
//			(void) sk_preempt_usec(10000);
		}
		my_thread_info->end_cycles = sk_cycles();
		//printf("%2lu.%12lu: End loop\n", cpuid, sk_cycles());
		async_msg = new sk_Msg;

		async_msg->data.p = NULL;
		sk_send_msg(my_thread_info->parent_pid, async_msg);
		//printf("%2lu.%12lu: Sent parent wakeup\n", cpuid, sk_cycles());
	}
}


static
void run_perf_test(size_t secs_to_run) {
	int i=0;
	setup_test();
	unsigned cur_package = UINT_MAX;

	thread_info_stats_t agg_stats;
	thread_info_stats_t pkg_stats[__CPU_SETSIZE];
	unsigned num_threads;
	perf_test_type_t test_type = counter32_test;
	unsigned lock_num = 0;
	unsigned lock_iteration = 0;
	unsigned run_number = 1;
	unsigned span_packages;
	for(test_type = counter64_test; test_type <= counter64_test; ) {
		for(num_threads = 1; num_threads <= num_cores_per_package; num_threads++) {
			for(lock_num = 0; lock_num < _countof(lock_array); lock_num++) {
				for(lock_iteration = 0; lock_iteration <= 1; lock_iteration++) {
					for(span_packages = 0; span_packages < 2; span_packages++) {
						cur_package = UINT_MAX;
						unsigned num_threads_remaining = num_threads;
						printf("%s Run %u, num_threads=%u, locking 0 vs %u\n", perf_test_type_str(test_type), run_number, num_threads, lock_num);
						i = 0;

						/*
						 * Start num_thread threads per package
						 */
						while(i >= 0 && i < lock_test_nvcpu()) {
							if(thread_infos[i].pkg_IDAPIC == cur_package) {
								if(num_threads_remaining > 0) {
									thread_infos[i].lock = (lock_iteration == 2) ? lock_array[(unsigned)i % _countof(lock_array)] : ((cur_package & 1) == (lock_iteration & 1)) ? lock_array[0] : lock_array[lock_num];
									ASSERT(thread_infos[i].lock != NULL);
									thread_infos[i].secs_to_run = secs_to_run;
									thread_infos[i].test_type = test_type;
									sk_Msg* async_msg = new sk_Msg;
									ASSERT(async_msg != NULL);

									async_msg->data.p = thread_infos[i].lock;
									sk_send_msg(&thread_infos[i], async_msg);
									num_threads_remaining--;
								}
								i++;
							} else {
								if(span_packages || cur_package == UINT_MAX) {
									cur_package = thread_infos[i].pkg_IDAPIC;
									num_threads_remaining = num_threads;
								} else {
									break;
								}
							}
						}

						/* Wait for threads to finish */
						for(i=0; i < lock_test_nvcpu(); i++) {
							if(thread_infos[i].secs_to_run) {
								sk_Msg *async_msg = (sk_Msg *)sk_receive_msg(&main_thread);
								delete async_msg;
							}
						}

						/* Print results */
						bzero(&agg_stats, sizeof(agg_stats));
						bzero(pkg_stats, sizeof(thread_info_stats_t) * num_packages);
						printf("         lock_addr    lock_cache_line     lock      cpu          ops        usecs        ops/s\n");
						for(i=0; i < lock_test_nvcpu(); i++) {
							if(thread_infos[i].end_cycles) {
								uint64_t cycles_elapsed = thread_infos[i].end_cycles - thread_infos[i].start_cycles;
								uint64_t usecs = cycles_elapsed * USECS_PER_SECOND / tsc_freq;
								printf("0x%016lx 0x%016lx       %2lu       %2u %12lu %12lu %12lu\n",
										(size_t)&thread_infos[i].lock->locks.counter32,
										((size_t)&thread_infos[i].lock->locks.counter32) >> 6,
										(((size_t)&thread_infos[i].lock->locks.counter32) >> 12) & 1,
										i,
										thread_infos[i].ops,
										usecs,
										thread_infos[i].ops * 1000000 / usecs);
								pkg_stats[thread_infos[i].pkg_IDAPIC].total_cycles += cycles_elapsed;
								pkg_stats[thread_infos[i].pkg_IDAPIC].total_ops += thread_infos[i].ops;

								agg_stats.total_cycles += cycles_elapsed;
								agg_stats.total_ops += thread_infos[i].ops;

							}
							thread_infos[i].end_cycles = thread_infos[i].start_cycles = thread_infos[i].ops = thread_infos[i].secs_to_run = 0;
						}
						printf("    package          ops        usecs        ops/s\n");
						for(i=0; i < (int)num_packages; i++) {
							uint64_t usecs = pkg_stats[i].total_cycles * USECS_PER_SECOND / tsc_freq;
							printf("         %02u %12lu %12lu %12lu\n",
									i,
									pkg_stats[i].total_ops,
									usecs,
									usecs ? pkg_stats[i].total_ops * 1000000 / usecs : 0);

						}
						uint64_t agg_usecs = agg_stats.total_cycles * USECS_PER_SECOND / tsc_freq;
						printf("Total       %12lu %12lu %12lu\n",
								agg_stats.total_ops,
								agg_usecs,
								agg_usecs ? agg_stats.total_ops * 1000000 / agg_usecs : agg_usecs);

						printf("\n");
						run_number++;
					}
				}
			}
		}
		test_type = (perf_test_type_t)(test_type + 1);
	}

}


static
void run_mem_perf_test(size_t secs_to_run) {
    int i;

    setup_test();
    if(!test_buffer) {
	ASSERT(num_pages > 1);
	ASSERT(buffer_size > 4096);
	test_buffer = (uint8_t*)malloc(buffer_size);
    }
    if(!test_buffer) {
	printf("Failed to allocate %lu byte buffer\n", buffer_size);
	return;
    } else {
	memset(test_buffer, 0,  buffer_size);
	printf("total memory: %lu, buffer_size: %lu, num pages=%lu, buffer: %p\n", memory_size, buffer_size, num_pages, test_buffer);

    }
    unsigned cur_package = UINT_MAX;

    thread_info_stats_t agg_stats;
    thread_info_stats_t pkg_stats[__CPU_SETSIZE];
    unsigned num_threads;
    perf_test_type_t test_type = counter32_test;
    unsigned run_number = 1;
    unsigned span_packages;
    for(test_type = memory_stride0; test_type <= memory_set_mix; ) {
	for(num_threads = 1; num_threads <= num_cores_per_package; num_threads++) {
	    for(span_packages = 0; span_packages < 2; span_packages++) {
		cur_package = UINT_MAX;
		unsigned num_threads_remaining = num_threads;
		printf("%s Run %u, num_threads=%u\n", perf_test_type_str(test_type), run_number, num_threads);
		i = 0;

		/*
		 * Start num_thread threads per package
		 */
		while(i >= 0 && i < lock_test_nvcpu()) {
		    if(thread_infos[i].pkg_IDAPIC == cur_package) {
			if(num_threads_remaining > 0) {
			    thread_infos[i].lock = NULL;
			    thread_infos[i].secs_to_run = secs_to_run;
			    thread_infos[i].test_type = test_type;
			    sk_Msg* async_msg = new sk_Msg;
			    ASSERT(async_msg != NULL);

			    async_msg->data.p = thread_infos[i].lock;
			    sk_send_msg(&thread_infos[i], async_msg);
			    num_threads_remaining--;
			}
			i++;
		    } else {
			if(span_packages || cur_package == UINT_MAX) {
			    cur_package = thread_infos[i].pkg_IDAPIC;
			    num_threads_remaining = num_threads;
			} else {
			    break;
			}
		    }
		}

		/* Wait for threads to finish */
		for(i=0; i < lock_test_nvcpu(); i++) {
		    if(thread_infos[i].secs_to_run) {
			sk_Msg *async_msg = (sk_Msg *)sk_receive_msg(&main_thread);
			delete async_msg;
		    }
		}

		/* Print results */
		bzero(&agg_stats, sizeof(agg_stats));
		bzero(pkg_stats, sizeof(thread_info_stats_t) * num_packages);
		printf("        cpu          ops        usecs        ops/s\n");
		for(i=0; i < lock_test_nvcpu(); i++) {
		    if(thread_infos[i].end_cycles) {
			uint64_t cycles_elapsed = thread_infos[i].end_cycles - thread_infos[i].start_cycles;
			uint64_t usecs = cycles_elapsed * USECS_PER_SECOND / tsc_freq;
			printf("         %2u %12lu %12lu %12lu\n",
				i,
				thread_infos[i].ops,
				usecs,
				thread_infos[i].ops * 1000000 / usecs);
			pkg_stats[thread_infos[i].pkg_IDAPIC].total_cycles += cycles_elapsed;
			pkg_stats[thread_infos[i].pkg_IDAPIC].total_ops += thread_infos[i].ops;

			agg_stats.total_cycles += cycles_elapsed;
			agg_stats.total_ops += thread_infos[i].ops;

		    }
		    thread_infos[i].end_cycles = thread_infos[i].start_cycles = thread_infos[i].ops = thread_infos[i].secs_to_run = 0;
		}
		printf("    package          ops        usecs        ops/s\n");
		for(i=0; i < (int)num_packages; i++) {
		    uint64_t usecs = pkg_stats[i].total_cycles * USECS_PER_SECOND / tsc_freq;
		    printf("         %02u %12lu %12lu %12lu\n",
			    i,
			    pkg_stats[i].total_ops,
			    usecs,
			    usecs ? pkg_stats[i].total_ops * 1000000 / usecs : 0);

		}
		uint64_t agg_usecs = agg_stats.total_cycles * USECS_PER_SECOND / tsc_freq;
		printf("Total       %12lu %12lu %12lu\n",
			agg_stats.total_ops,
			agg_usecs,
			agg_usecs ? agg_stats.total_ops * 1000000 / agg_usecs : agg_usecs);

		printf("\n");
		run_number++;
	    }


	}
	test_type = (perf_test_type_t)(test_type + 1);
    }


}

static void
memperf_usage(void)
{
	printf("usage: memperf -p <run time in secs>\n");
}


int main(int argc, char **argv) {
    int ch;
    size_t secs_to_run = 10;
	sk_init_cycles();
	setup_test();
	while ((ch = getopt(argc, argv, "p:")) != -1) {
		switch(ch) {
		case 'p':
			secs_to_run = strtoul(optarg, NULL, 10);
                        break;
		default:
			memperf_usage();
			return 1;
		};
	}
	run_perf_test(secs_to_run);
	run_mem_perf_test(secs_to_run);
	return 0;
}
