---
layout: post
title:  How to Destroy a Team 
date:   2025-08-17 21:07:39 -0400
categories: rant
tags: rant
---

> reference [3 Pillars in Debugging Teams](https://book.debuggingteams.com/#the_three_pillars)

If you want to get to demolishing a team of productive devs asap, just read these and get to it. Details don't matter anyway, shift the blame when things don't work out!

1. Give commands, don't reach consensus

	A Senior developer is just an intern that types faster! Give them prescriptive tasks, don't let them design systems to fill customer requests. Software development is not a collaboration between customer needs and technical realities. Individual contributors should just do what you say, after all, YOU are the MANAGER of the PRODUCT. The developer's job is not to give feedback, it's to implement exactly what you say.

2. Embrace Complication

	The real world is complicated, try to model it exactly. Do not design for the 95% of cases

3. Tell, don't ask, the customer (developers) what they want

	Developers ask for boring things, like simple systems that work reliably. Instead, sell them on cool whizbangs and sweep core functionality under the rug. Think of a cool idea, and pitch it until someone doesn't say "no." Don't make a high level Product Requirements Document, get feedback, refine, and share it with the developers. If everyone knows the plan, then your usefulness will be over, so jealously guard the next sprint's work until the planning meeting.

4. If you get pushback, go around!

	Use political maneuvering whenever you get resistance. If an objection is raised, say "well this was already decided, you should disagree and commit!" If they insist, stall out the meeting, then go to other managers and get them all to side with you on a fun fun fun 4 on 1 call. Ignore the plebeian developer, they do not understand the product.

Do these things, and you will make your senior engineers so exasperated that they will surely leave, and the team will collapse.

## The actual story

Yes, this is only a thin guise of insightful tips in order to slip in a rant like the cheese containing a dewormer pill given to a beloved but obstinate family pet.

### The setup

I was working on the "DevOps" team which was composed of smaller teams handling Build, Test, and Release systems. All of which were done in-house due to weird constraints that were difficult to work around with out-of-house solutions in a cost effective way (lol, I'll rant on that later.) Simple enough, it took events from GitHub, built and packaged with docker, exported artifacts, ran unit and integration test suites, e2e tests, then deployed and promoted through our environmentshttps://notgoodatcoding.github.io/rant/2025/08/18/how-to-destroy-a-team.html (dev, staging, pre-prod, prod, global zones.)

### Enter complications

Around the time PM joined, we were planning on unifying the Build/Test/Release systems into one general workflow orchestrator with Argo. We had just gone over the technical reasons, limitations, and special cases that made commodity CI/CD untenable. The in-house separate systems were slowing down, making integration more and more difficult. They were complicated, not particularly complex, and that distinction will be important for this whole ~rant~ story. Classic software, more and more dev resources were just going to maintenance. So, our architect and tech lead (and a few senior devs, myself included) hashed out a plan for how to get things up to feature pairity, and how future integration would work. All was good.

I didn't hear anything from ProductManager for weeks, then suddenly we all had a meeting where we were told what we were going to be doing for the next year. There was the original plan, and there on top of it were a slew of additional boxes, like some strange LinkedIn graffiti tag. We were railroaded into implementing a complete hierarchical configuration management system on top of the CI/CD system, and that would configure the control flow at every level of CI/CD with overrides and defaults. This would be a fully self-contained org/team/techlead/role4 representation of the **ENTIRE DEV ORG** (which, honestly was not that large, only 800 or so at the time, I believe.)

> The complicated spaghetti wasn't creeping in, it was already on the table before the tablecloth was even placed.

PM proudly explained this system, one that had only the least bit of an oblique relation to building, testing, or releasing company code. We would have to create all these teams and roles, set up and maintain all of the org -> group -> team -> repository relations, as well as making the defaults and overrides logically correct. Some controls would override from org down to repo, some would not. In the real world, this is how the dev org looked and therefore, that's how the DB would look. 

### When you can't go through it and you can't knock it down, you know that you've found

I asked TechLead about this, they described long meetings with a [brick wall](https://www.youtube.com/watch?v=oXp8Th41rBs). ProductManager wouldn't budge at all. For hours. Before meeting with TechLead, ProductManager had already convinced 3 EngineeringManagers (another issue, another rant) of this, and had them backing up the plan like cartoon henchmen. None of the core functionality was even started in earnest, we only had the start of a Proof of Concept. But worry not, surely this can be implemented soon enough, and we can get back to work. ProductManager will get their feeling of "yay I did something :DDD" and with just a little splinter pulling, we lowly devs could get back to real work.

It should have been done soon enough, maybe with a bit more technical skill from myself, I could have done it faster. But soon I found myself in a cycle of reviewing changes with so many issues that the choice was to write it myself in the comments or write it again in a follow up change. I realize this sounds awfully pretentious, but I was not the only one that was doing this and felt this way. At first. We simply did not have the build up of skills on the team. Some were completely ossified in the system they had been working on for their entire career. We also had a lot of interns and not enough mentors. So we slogged on. I split my time between intensive reviews on this system and the main workflow orchestrator. It was going excruciatingly slow for about 8-9 months, with frustrations slowly building.

### This is the part I pretend I learned something

This eroded the [Three Pillars](https://book.debuggingteams.com/#the_three_pillars)

#### Humility
ProductManager refused to yield, their ideas were not open to discussion, unless the discussion was about when the work would be done. They only wanted to hear about getting the pet features in, not the core product. This led to ego vs. ego trying to butt heads to get anywhere.

#### Respect 
Our whole dev team felt completely disrespected by ProductManager, betrayed by our EngineeringManagers, and left to dry by TechLead. We had a lot wrapped up in this work, most of the senior team members had been working on this problem area over their entire careers and now their opinions were not even asked for, and when they spoke up those insights were trashed. This stung especially because of how quickly the team was taken over by ProductManager. We were steamrolled, and then ignored.

#### Trust
This broke when ProductManger first sprung the "well, all the managers agree so you should just do it." I could see the shift immediately. There was no discussion with us, it was purposefully hidden so that ProductManager could stack the deck before the first hand. Further, it became adversarial (you can tell just by the metaphor.) We knew what we were doing, we knew our customers because we worked with them for years. We knew that B liked Gradle builds more than Maven, so we added support explicitly. We knew all the pain points because we had felt them. Then, we weren't asked once for any input. How can you trust in that environment?

### Mr. Gorbachev, let's find somewhere else to ride our bikes

Things started to slip. We worked on the (doomed) org hierarchy system at the expense of the main Build/Test/Release work. Things slowed down, and when the pet started slowing down, more dev time was forced to it. Two senior developers left, leaving myself, another 2 "lifer" seniors, and TechLead. Then TechLead dropped, claiming that it was just too much dealing with ProductManager. By the end of a year, we had the simplest static workflow running for about 1/3 of all repos. Those were the easiest ones, expected to be maybe 10% of the work to configure. I thought that maybe it was true that with a little more work, we could get things back on track. I started working more, after hours, making low estimates, doing untracked work. Other team members detached in other ways, picking at the simplest bugs like a mushy pea on a non-brit's plate. Eventually even my work that was under-estimated in points was taking up 60, creeping up on 80% of the total ticket throughput. I decided enough was enough and transferred to another team. Somewhat toxically I periodically checked on the burndowns after leaving, and they were flat.

I cared about the build system. I cared even after I left the team. I cared after I left the company. I care now, writing this. In a way, the destruction of that project and team even led me to a much better career path. When things fall apart, some people make it out OK. I heard that ProductManger was fired a few months after I took a new job, but the team lost, I think, 5 good engineers (no, I don't include myself in that.) The project was abandoned after a total of ~12 devs working for 14 months on it, and instead they went with G_____ A______ which the DevOps team manually set up for all of the hundreds of repos for the org.

As disappointing as that is, it's a better trade according to the [Universal Law of Automation](https://xkcd.com/1205/)


